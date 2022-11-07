# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####


bl_info = {
	"name": "Motion Trail (update)",
	"author": "Bart Crouch, Viktor_smg",
	"version": (0, 16, 0),
	"blender": (3, 2, 0),
	"location": "View3D > Toolbar > Motion Trail tab",
	"warning": "Support for features not originally present is buggy; NO UNDO!!!",
	"description": "Display and edit motion trails in the 3D View",
	"category": "Animation",
}

from typing import List
import gpu
from gpu_extras.batch import batch_for_shader
import blf
import bpy
from bpy_extras import view3d_utils
import math
import mathutils
from bpy.props import (
		BoolProperty,
		EnumProperty,
		FloatProperty,
		FloatVectorProperty,
		IntProperty,
		StringProperty,
		PointerProperty,
		)
		
import sys
import traceback
from functools import reduce
from collections.abc import Callable

from bpy.types import Object, PoseBone, Context, Action, FCurve, Keyframe
from mathutils import Matrix, Vector, Quaternion, Euler

# Linear interpolation for 4-element tuples
def lerp4(fac, tup1, tup2):
	return (* [tup1[i] * fac + tup2[i]*(1.0-fac) for i in range(4)],)

def add4(tup1, tup2):
	return (* [tup1[i] + tup2[i] for i in range(4)],)

def mul4(tup1, tup2):
	return (* [tup1[i]*tup2[i] for i in range(4)],)

def mulscalar(tup, scalar):
	return (* [tup[i]*scalar for i in range(4)],)

def make_chan(i):
	return tuple([j == i for j in range(3)])

# Flattens recursively.
def flatten(deeplist):
	flatlist = []
	for elem in deeplist:
		if type(elem) is list:
			flatlist.extend(flatten(elem))
		else:
			flatlist.append(elem)
	return flatlist

# fake fcurve class, used if no fcurve is found for a path
class fake_fcurve():
	def __init__(self, object: Object | PoseBone, index, rotation=False, scale=False):
		# location
		if not rotation and not scale:
			self.loc = object.location[index]
		# scale
		elif scale:
			self.loc = object.scale[index]
		# rotation
		elif rotation == 'QUATERNION':
			self.loc = object.rotation_quaternion[index]
		elif rotation == 'AXIS_ANGLE':
			self.loc = object.rotation_axis_angle[index]
		else:
			self.loc = object.rotation_euler[index]
		self.keyframe_points = []

	def evaluate(self, frame):
		return(self.loc)

	def range(self):
		return([])


class MatrixCache():
	__mats: dict[(float, Object|PoseBone), (Matrix, Vector, Quaternion, Vector)]
	getter: Callable[[float, Object|PoseBone, Context], Matrix]

	def __init__(self, _getter: Callable[[float, Object|PoseBone, Context], Matrix]):
		self.__mats = {}
		self.getter = _getter

	def __build_entry(self, frame, obj, context):

		mat: Matrix = self.getter(frame, obj, context)
		decomposed = mat.decompose()
		self.__mats[(frame, obj)] = (mat, *decomposed)

	def __guarantee_entry(self, frame, obj, context):
		if not (frame, obj) in self.__mats:
			self.__build_entry(frame, obj, context)

	def get_matrix(self, frame, obj, context) -> Matrix:
		self.__guarantee_entry(frame, obj, context)
		return self.__mats[(frame, obj)][0]

	def get_location(self, frame, obj, context) -> Vector:
		self.__guarantee_entry(frame, obj, context)
		return self.__mats[(frame, obj)][1]

	def get_rotation(self, frame, obj, context) -> Quaternion:
		self.__guarantee_entry(frame, obj, context)
		return self.__mats[(frame, obj)][2]

	def get_scale(self, frame, obj, context) -> Vector:
		self.__guarantee_entry(frame, obj, context)
		return self.__mats[(frame, obj)][3]

	def get_tuple(self, frame, obj, context):
		self.__guarantee_entry(frame, obj, context)
		return [self.__mats[(frame, obj)][i] for i in range(1, 3)]


def get_curves_action(obj: Object | PoseBone, action: Action) -> List[List[FCurve]]:
	""" Get f-curves for [[loc], [rot], [scale]] from an Object or PoseBone and an associated action. Rotation fcurves may be 4 if quaternion is used."""
	locpath = obj.path_from_id("location")
	rotpath = ""
	
	quat = True
	if obj.rotation_mode == 'QUATERNION':
		rotpath = obj.path_from_id("rotation_quaternion")
	else:
		rotpath = obj.path_from_id('rotation_euler')
		quat = False
	sclpath = obj.path_from_id("scale")

	rotrange = 4
	if not quat:
		rotrange = 3
	
	loccurves = [action.fcurves.find(locpath, index=i) for i in range(3)]
	rotcurves = [action.fcurves.find(rotpath, index=i) for i in range(rotrange)]
	sclcurves = [action.fcurves.find(sclpath, index=i) for i in range(3)]

	curves = [loccurves, rotcurves, sclcurves]
	curves_fakes = [(False, False), (obj.rotation_mode, False), (False, True)]
	curves_ranges = [3, rotrange, 3]

	for i in range(3):
		for j in range(curves_ranges[i]):
			if curves[i][j] is None:
				curves[i][j] = fake_fcurve(obj, j, curves_fakes[i][0], curves_fakes[i][1])
	
	return curves

def get_curves(obj: Object | PoseBone):
	"""Get f-curves for [[loc], [rot], [scale]] from an Object or PoseBone and its default action. Rotation fcurves may be 4 if quaternion is used. Returns [] if no action."""
	animDataContainer = obj
	if type(obj) is PoseBone:
		animDataContainer = obj.id_data

	if not animDataContainer.animation_data.action:
		return []
	
	return get_curves_action(obj, animDataContainer.animation_data.action)

# turn screen coordinates (x,y) into world coordinates vector
def screen_to_world(context, x, y):
	depth_vector = view3d_utils.region_2d_to_vector_3d(
							context.region, context.region_data, [x, y]
							)
	vector = view3d_utils.region_2d_to_location_3d(
							context.region, context.region_data, [x, y],
							depth_vector
							)

	return(vector)


# turn 3d world coordinates vector into screen coordinate integers (x,y)
def world_to_screen(context, vector):
	prj = context.region_data.perspective_matrix @ \
		Vector((vector[0], vector[1], vector[2], 1.0))
	width_half = context.region.width / 2.0
	height_half = context.region.height / 2.0

	x = int(width_half + width_half * (prj.x / prj.w))
	y = int(height_half + height_half * (prj.y / prj.w))

	# correction for corner cases in perspective mode
	if prj.w < 0:
		if x < 0:
			x = context.region.width * 2
		else:
			x = context.region.width * -2
		if y < 0:
			y = context.region.height * 2
		else:
			y = context.region.height * -2

	return(x, y)


def get_matrix_frame(obj: Object | PoseBone, frame, action):
	""" Get a LocRotScale matrix assembled from the respective f-curves for a given frame, for the object or posebone and the given action."""

	curves = get_curves_action(obj, action)
	
	loc = Vector([c.evaluate(frame) for c in curves[0]])
	if obj.rotation_mode == 'QUATERNION':
		rot = Quaternion([c.evaluate(frame) for c in curves[1]])
	else:
		rot = Euler([c.evaluate(frame) for c in curves[1]])
	scale = Vector([c.evaluate(frame) for c in curves[2]])
	
	return Matrix.LocRotScale(loc, rot, scale)
	
# Get the world-ish matrix for an object, factoring in its parents recursively, if any
def get_matrix_obj_parents(obj, frame, do_anim=True):
	mat = None
	
	if do_anim:
		mat = get_matrix_frame(obj, frame, obj.animation_data.action)
	else:
		mat = Matrix()

	parentMat = obj.matrix_parent_inverse
	if obj.parent:
		parentMat = get_matrix_obj_parents(obj.parent, frame) @ parentMat
		
	res = parentMat @ mat
	
	if obj.constraints:
		res = evaluate_constraints(res, obj.constraints, frame, obj)
		
	return res

# Get the armature space matrix for a bone
def get_matrix_bone_parents_as(pose_bone, frame, do_anim = True):
	animMat = None
	ob = pose_bone.id_data
	
	if do_anim:
		animMat = get_matrix_frame(pose_bone, frame, ob.animation_data.action)
	else:
		animMat = Matrix()
		
	parentMat = None
	parentOffsetMat = None
	if pose_bone.parent:
		parentMat = get_matrix_bone_parents_as(pose_bone.parent, frame)
		parentOffsetMat = pose_bone.parent.bone.matrix_local.inverted() @ pose_bone.bone.matrix_local
	else:
		parentMat = Matrix()
		parentOffsetMat = pose_bone.bone.matrix_local
		
	res = parentMat @ parentOffsetMat @ animMat
		
	if pose_bone.constraints:
		res = evaluate_constraints(res, pose_bone.constraints, frame, pose_bone)
	
	return res

def get_matrix_bone_parents(pose_bone, frame, do_anim = True):
	return get_matrix_obj_parents(pose_bone.id_data, frame) @ \
	get_matrix_bone_parents_as(pose_bone, frame, do_anim)

# Get the world-ish matrix of a bone or object
def get_matrix_any_custom_eval(frame: float, thing: Object | PoseBone, do_anim = True) -> Matrix:
	if type(thing) is PoseBone:
		return get_matrix_bone_parents(thing, frame, do_anim)
	return get_matrix_obj_parents(thing, frame, do_anim)

# Get matrix for child of constraint
def evaluate_childof(constraint, frame):
	mat = Matrix()
	try:
		if constraint.subtarget:
			mat = get_matrix_bone_parents(constraint.target.\
			pose.bones[constraint.subtarget], frame)
		else:
			mat = get_matrix_obj_parents(constraint.target, frame)
			
		mat = mat @ constraint.inverse_matrix
			
		bool_names = ["use_location_x", "use_location_y", "use_location_z", "use_rotation_x", "use_rotation_y",
		"use_rotation_z", "use_scale_x", "use_scale_y", "use_scale_z"]
		bools = [getattr(constraint, bool_names[i]) for i in range(len(bool_names))]
		if not reduce((lambda a, b: a and b), bools, True):
			zeros = [0, 0, 0, 0, 0, 0, 1, 1, 1]
			(disassembledLoc, disassembledRot, disassembledScl) = mat.decompose()
			for i in range(3):
				if not bools[i]:
					disassembledLoc[i] = zeros[i]
			for i in range(3, 6):
				if not bools[i]:
					disassembledRot[i-3] = zeros[i]
			for i in range(6, 9):
				if not bools[i]:
					disassembledScl[i-6] = zeros[i]
			mat = Matrix.LocRotScale(disassembledLoc, disassembledRot, disassembledScl)
	
	except Exception as e:
		print(e)
		tb = sys.exc_info()[-1]
		print(traceback.extract_tb(tb))
	
	finally:
		return mat
			
constraint_funcs = {'CHILD_OF': evaluate_childof}

# Get matrices from all constraints?
def evaluate_constraints(mat, constraints, frame, ob):
	accumulatedMat = Matrix()
	for c in constraints:
		f = constraint_funcs.get(c.type)
		if f is None or not c.enabled or c.influence == 0.0:
			continue
		constraintMat = f(c, frame)
		if c.influence != 1.0:
			constraintMat = constraintMat.lerp(Matrix(), 1.0-c.influence)
		accumulatedMat = accumulatedMat @ constraintMat
	return accumulatedMat @ mat

def get_matrix_any_depsgraph(frame: float, target: Object | PoseBone, context: Context) -> Matrix:
	oldframe = context.scene.frame_float
	context.scene.frame_float = frame

	dg = context.evaluated_depsgraph_get()
	
	isBone = type(target) is PoseBone
	ob = target.id_data if isBone else target
		
	evalledOb = ob.evaluated_get(dg)

	if isBone:
		resMat = evalledOb.matrix_world @ evalledOb.pose.bones[target.name].matrix
	else:
		resMat = evalledOb.matrix_world

	context.scene.frame_float = oldframe
	# TODO: When updates are forced with DG on, playback can freeze... This may be due to setting the frame in the DG functions. Is this fixable?
	return resMat

# Calculate an inverse matrix for an object or bone, such that it's suitable for the addon's
# manipulation of keyframes (IE without the very last animation applied)
# using our own, draw handler-safe methods
def get_inverse_parents(frame, ob, context):
	return get_matrix_any_custom_eval(ob, frame, False).inverted()

def get_inverse_parents_depsgraph(frame, ob, context):
	mat = ''
	if type(ob) is PoseBone:
		mat = get_matrix_frame(ob, frame, ob.id_data.animation_data.action)
	else:
		mat = get_matrix_frame(ob, frame, ob.animation_data.action)
		
	return (get_matrix_any_depsgraph(frame, ob, context) @ mat.inverted()).inverted()

def get_original_animation_data(context, keyframes: dict[(any, List[float])], cache: MatrixCache):
	"""Get position of keyframes and handles at the start of dragging.\n
	Returns keyframes_ori: {ob: [chan: [fcurve: {frame: ([x, y] x3), ...} x 3/4?] x3?], ...},\n
	where the 3 Vectors are the coordinates of the keyframe, its left handle, and its right handle.\n
	The keyframe's frame will be duplicated as it's the key and the 1st coordinate, but no biggie."""
	keyframes_ori = {}

	if context.active_object and context.active_object.mode == 'POSE':
		objects = [pb for pb in context.selected_pose_bones]
	else:
		objects = [ob for ob in context.selected_objects]

	for ob in objects:
		curves = get_curves(ob)
		if len(curves) == 0:
			continue

		# TODO: Should a raw PB/Object be used as a dict key?
		keyframes_ori[ob] = [[], [], []]
		for chan in range(len(curves)):
			for fcurv in range(len(curves[chan])):
				keyframes_ori[ob][chan].append([])
				kf: Keyframe
				for kf in curves[chan][fcurv].keyframe_points:
					keyframes_ori[ob][chan][fcurv][kf.co[0]] = (kf.co.copy(), kf.handle_left.copy(), kf.handle_right.copy())

	return keyframes_ori

def merge_items(enum1, enum2, mergec = 3):
	"""Merge 2 sorted lists of structure [[frame, stuff, [bool, bool, bool, ... mergec times]], ...]"""

	def mergetruth(l1, l2):
		return tuple([l1[i] or l2[i] for i in range(mergec)])

	i, j = 0, 0
	res = []
	while i<len(enum1) and j<len(enum2):
		if enum1[i][0] == enum2[j][0]:
			indices1 = enum1[i][1][1]
			indices2 = enum2[i][1][1]
			indices_merged = mergetruth(indices1, indices2)
			res.append([enum1[i][0], [enum1[i][1][0], indices_merged]])
			i += 1
			j += 1
			continue
		if enum1[i][0] < enum2[j][0]:
			res.append([enum1[i][0], enum1[i][1]])
			i += 1
		else:
			res.append([enum2[j][0], enum2[j][1]])
			j += 1
						
	while i<len(enum1):
		res.append([enum1[i][0], enum1[i][1]])
		i += 1

	while j<len(enum2):
		res.append([enum2[j][0], enum2[j][1]])
		j += 1

	return res

def merge_dicts(dict_list):
	"""Merge dicts in a list, with structure [Dict(frame, [stuff, [bool, bool, ...]])] into a single Dict(frame, [stuff, [bool, bool, bool]])"""

	itemized = [list(a_dict.items()) for a_dict in dict_list]

	merged_lists = merge_items(itemized[0], merge_items(itemized[1], itemized[2]))
	final_dict = {}
	for elem in merged_lists:
		final_dict[elem[0]] = elem[1]

	return final_dict

# callback function that calculates positions of all things that need be drawn
def calc_callback(self, context, inverse_getter, matrix_getter):
	# Remove handler if file was changed and we lose access to self
	# I'm all ears for a better solution, as __del__ for the modal operator does not call on file change
	# and there is no special event emitted to the operator for that
	# (besides "TIMER" which gets emitted other not so opportune times as well)
	# Also, this might have the tendency to get called twice? So, that's what the extra if is for.
	try:
		self.properties
	except:
		if global_mtrail_handler_calc:
			bpy.types.SpaceView3D.draw_handler_remove(global_mtrail_handler_calc, 'WINDOW')
		return
	
	mt: MotionTrailProps = context.window_manager.motion_trail

	if context.active_object and context.active_object.mode == 'POSE':
		objects = [pb for pb in context.selected_pose_bones]
	else:
		objects = [ob for ob in context.selected_objects]

	if objects == self.displayed:
		selection_change = False
	else:
		selection_change = True

	if self.lock and not selection_change and \
	context.region_data.perspective_matrix == self.perspective and not \
	mt.force_update and self.last_frame == context.scene.frame_float:
		return

	self.last_frame = context.scene.frame_float

	# dictionaries with key: objectname
	self.paths = {} 	           # value: list of lists with x, y, color and frame
	self.keyframes = {}            # value: dict with frame as key and [x,y] as value
	self.handles = {}    # value: {ob: [{frame: {"left": co, "right":co}, ...} x3], ...}
	self.timebeads = {}            # value: dict with frame as key and [x,y] as value
	self.click = {} 	           # value: list of lists with frame, type, loc-vector
	self.spines = {}               # value: dict with frame as key and [x0,y0, [(x1, y1), (x2,y2), ...]] as values, for 1..6 where xy1,2,3 = +x,+y,+z and x4,5,6 = -x,-y,-z
	if selection_change:
		# value: editbone inverted rotation matrix or None
		self.active_keyframe = False
		self.active_handle = False
		self.active_timebead = False
		self.active_frame = False
		self.highlighted_coord = False
	
	if selection_change or not self.lock or mt.force_update:
		self.cache = MatrixCache(matrix_getter)

	self.perspective = context.region_data.perspective_matrix.copy()
	self.displayed = objects  # store, so it can be checked next time
	mt.force_update = False
	try:
		#global_undo = context.preferences.edit.use_global_undo
		#context.preferences.edit.use_global_undo = False
		for ob in objects:
			curves = get_curves(ob)
			if len(curves) == 0:
				continue

			scene = context.scene
			if mt.path_before == 0:
				range_min = scene.frame_start
			else:
				range_min = max(scene.frame_start, scene.frame_current - mt.path_before)

			if mt.path_after == 0:
				range_max = scene.frame_end
			else:
				range_max = min(scene.frame_end, scene.frame_current + mt.path_after)

			# get location data of motion path
			path = []
			speeds = []
			step = mt.path_step

			prev_loc = self.cache.get_location(range_min - 1, ob, context)
			for frame in range(range_min, range_max + 1, step):
				loc = self.cache.get_location(frame, ob, context)
				if not context.region or not context.space_data:
					continue
				x, y = world_to_screen(context, loc)
				if mt.path_style == 'simple':
					path.append([x, y, [0.0, 0.0, 0.0, 1.0], frame])
				else:
					dloc = (loc - prev_loc).length
					path.append([x, y, dloc, frame])
					speeds.append(dloc)
					prev_loc = loc
			# calculate color of path
			if mt.path_style == 'speed':
				speeds.sort()
				min_speed = speeds[0]
				d_speed = speeds[-1] - min_speed
				d_speed = max(d_speed, 1e-6)
				for i, [x, y, d_loc, frame] in enumerate(path):
					relative_speed = (d_loc - min_speed) / d_speed # 0.0 to 1.0
					fac = min(1.0, 2.0 * relative_speed)
					path[i][2] = lerp4(fac, mt.speed_color_max, 
					mt.speed_color_min)
			elif mt.path_style == 'acceleration':
				accelerations = []
				prev_speed = 0.0
				for i, [x, y, d_loc, frame] in enumerate(path):
					accel = d_loc - prev_speed
					accelerations.append(accel)
					path[i][2] = accel
					prev_speed = d_loc
				accelerations.sort()
				min_accel = accelerations[0]
				max_accel = accelerations[-1]
				for i, [x, y, accel, frame] in enumerate(path):
					if accel < 0:
						relative_accel = accel / min_accel  # values from 0.0 to 1.0
						fac = 1.0 - relative_accel
						path[i][2] = lerp4(fac, mt.accel_color_neg, 
						mt.accel_color_static)
					elif accel > 0:
						relative_accel = accel / max_accel  # values from 1.0 to 0.0
						fac = 1.0 - relative_accel
						path[i][2] = lerp4(fac, mt.accel_color_static, 
						mt.accel_color_pos)
					else:
						path[i][2] = mt.accel_color_static
			self.paths[ob] = path



			# get keyframes and handles
			keyframes = [{}, {}, {}]
			handle_difs = [{}, {}, {}]
			kf_time = [[], [], []]
			click = []

			# TODO: should this be called "categories"?
			channels = (mt.do_location, mt.do_rotation, mt.do_scale)

			for chan in range(3):
				if not channels[chan]: 
					continue

				quat = len(curves[chan]) == 4

				for fc in curves[chan]:
					for kf in fc.keyframe_points:
						# handles for values mode
						if mt.mode == "values":
							if kf.co[0] not in handle_difs[chan]:
								handle_difs[chan][kf.co[0]] = {"left": Vector(), "right": Vector()}
									
							if not quat:
								ldiff = Vector(kf.handle_left[:]) - Vector(kf.co[:])
								rdiff = Vector(kf.handle_right[:]) - Vector(kf.co[:])


								hdir = mt.handle_direction
								lco = 0.0
								rco = 0.0
								
								if hdir == 'time':
									lco = ldiff.normalized()[1]
									rco = rdiff.normalized()[1]
								elif hdir == 'wtime':
									lco = sum(ldiff.normalized() * Vector((0.25, 0.75)))
									rco = sum(rdiff.normalized() * Vector((0.25, 0.75)))
								elif hdir == 'value':
									lco = ldiff.normalized()[0]
									rco = rdiff.normalized()[0]
								elif hdir == 'wloc':
									lco = sum(ldiff.normalized() * Vector((0.75, 0.25)))
									rco = sum(rdiff.normalized() * Vector((0.75, 0.25)))
								elif hdir == 'len':
									lco = -ldiff.length
									rco = rdiff.length
								
								handle_difs[chan][kf.co[0]]["left"][fc.array_index] = lco
								handle_difs[chan][kf.co[0]]["right"][fc.array_index] = rco

							else:
								# !! This code running multiple times might sound bad, but consider the worse scenario in which someone shifted a single quaternion keyframe. This handles it.
								rot = self.cache.get_rotation(kf.co[0], ob, context)
								vec = mathutils.Vector((0.0, 0.0, 1.0)) # TODO: think of better vector to represent rotation with?
								vec.rotate(rot)

								handle_difs[chan][kf.co[0]]["left"] = vec
								handle_difs[chan][kf.co[0]]["right"] = -vec

						# keyframes
						if kf.co[0] in kf_time[chan]:
							continue
						kf_time[chan].append(kf.co[0])
						kf_frame = kf.co[0]

						loc = self.cache.get_location(kf_frame, ob, context)
						x, y = world_to_screen(context, loc)
						keyframes[chan][kf_frame] = [[x, y], make_chan(chan)]
				lasti = chan

			if sum(channels) <= 1:
				self.keyframes[ob] = keyframes[lasti]
			else:
				self.keyframes[ob] = merge_dicts(keyframes)

			if mt.mode != 'speed':
				# can't select keyframes in speed mode
				for kf_frame, [coords, kf_channels] in self.keyframes[ob].items():
					click.append( [kf_frame, "keyframe", Vector(coords), kf_channels] )

			# handles are only shown in value-altering mode
			if mt.mode == 'values' and mt.handle_display:
				# calculate handle positions
				handles = [{}, {}, {}]

				for chan in range(3):
					if not channels[chan]: 
						continue

					for frame, vecs in handle_difs[chan].items():

						# Back to world space?
						mat = inverse_getter(frame, ob, context)
						vec_left = vecs["left"] @ mat
						vec_right = vecs["right"] @ mat
							
						hlen = mt.handle_length
						vec_left = vec_left * hlen
						vec_right = vec_right * hlen
						vec_keyframe = self.cache.get_location(frame, ob, context)

						x_left, y_left = world_to_screen(context, vec_left * 2 + vec_keyframe)
						x_right, y_right = world_to_screen(context, vec_right * 2 + vec_keyframe)

						handles[chan][frame] = {"left": [x_left, y_left], "right": [x_right, y_right]}

						click.append([frame, "handle_left", Vector([x_left, y_left]), make_chan(chan)])
						click.append([frame, "handle_right", Vector([x_right, y_right]), make_chan(chan)])
				
				self.handles[ob] = handles # !! Handles are stored unmerged.

			# calculate timebeads for timing mode
			if mt.mode == 'timing':
				timebeads = {}
				n = mt.timebeads * (len(kf_time[0]) + len(kf_time[1]) + len(kf_time[2]) - 1) # TODO: Is this correct?
				dframe = (range_max - range_min) / (n + 1)

				for i in range(1, n + 1):
					frame = range_min + i * dframe
					loc = self.cache.get_location(frame, ob, context)
					x, y = world_to_screen(context, loc)
					timebeads[frame] = [[x, y], channels]
					click.append( [frame, "timebead", Vector([x, y]), make_chan(chan)] )
				self.timebeads[ob] = timebeads

			# calculate timebeads for speed mode
			if mt.mode == 'speed':
				timebead_container = [{}, {}, {}]
				lasti = 0

				for chan in range(3):
					if not channels[chan]: 
						continue

					angles = dict([[kf_frame, {"left": [], "right": []}] for kf_frame, [kf, kf_channels] in keyframes[chan].items()])
					for fc in curves[chan]:
						for i, kf in enumerate(fc.keyframe_points):
							if i != 0:
								angle = Vector([-1, 0]).angle(
													Vector(kf.handle_left) -
													Vector(kf.co), 0
													)
								if angle != 0:
									angles[kf.co[0]]["left"].append(angle)
							if i != len(fc.keyframe_points) - 1:
								angle = Vector([1, 0]).angle(
													Vector(kf.handle_right) -
													Vector(kf.co), 0
													)
								if angle != 0:
									angles[kf.co[0]]["right"].append(angle)
					timebeads = {}
					kf_time[chan].sort()
					for frame, sides in angles.items():
						if sides["left"]:
							perc = (sum(sides["left"]) / len(sides["left"])) / \
								(math.pi / 2)
							perc = max(0.4, min(1, perc * 5))
							previous = kf_time[chan][kf_time[chan].index(frame) - 1]
							bead_frame = frame - perc * ((frame - previous - 2) / 2)

							loc = self.cache.get_location(bead_frame, ob, context)

							x, y = world_to_screen(context, loc)
							timebeads[bead_frame] = [[x, y], channels]
						if sides["right"]:
							perc = (sum(sides["right"]) / len(sides["right"])) / \
								(math.pi / 2)
							perc = max(0.4, min(1, perc * 5))
							next = kf_time[chan][kf_time[chan].index(frame) + 1]
							bead_frame = frame + perc * ((next - frame - 2) / 2)

							loc = self.cache.get_location(bead_frame, ob, context)

							x, y = world_to_screen(context, loc)
							timebeads[bead_frame] = [[x, y], make_chan(chan)]

					timebead_container[chan] = timebeads
					lasti = chan

				if sum(channels) <= 1:
					self.timebeads[ob] = timebead_container[lasti]
				else:
					self.timebeads[ob] = merge_dicts(timebead_container)

				for bead_frame, [coords, bead_channels] in self.timebeads[ob].items():
					click.append( [bead_frame, "timebead", Vector(coords), bead_channels] )

			if mt.show_spines:
				for frame in range(range_min, range_max + 1, mt.spine_step):
					loc = self.cache.get_location(frame, ob, context)
					if mt.spine_do_rotation:
						rot = self.cache.get_rotation(frame, ob, context)
					else:
						rot = Euler()

					if mt.spine_do_scale:
						scl = self.cache.get_scale(frame, ob, context)
					else:
						scl = Vector((1.0, 1.0, 1.0))

					baseLoc = world_to_screen(context, loc)

					slen = mt.spine_length

					resLocs = []
					vecs = ((slen * scl[0], 0, 0), (0, slen * scl[1], 0), (0, 0, slen * scl[2]), (-slen * scl[0], 0, 0), (0, -slen * scl[1], 0), (0, 0, -slen * scl[2]))
					for i in range(6):
						vec = Vector(vecs[i])
						vec.rotate(rot)
						vec.rotate(mt.spine_offset) # Is this slow enough to warrant an if?
						resLocs.append(world_to_screen(context, loc + vec))
					
					self.spines[frame] = (baseLoc, resLocs)

			# add frame positions to click-list
			if mt.frame_display:
				path = self.paths[ob]
				for x, y, color, frame in path:
					click.append( [frame, "frame", Vector([x, y]), channels] )

			self.click[ob] = click

		#context.preferences.edit.use_global_undo = global_undo

	except Exception as e:
		print(e)
		tb = sys.exc_info()[-1]
		print(traceback.extract_tb(tb))
		# restore global undo in case of failure (see T52524)
		#context.preferences.edit.use_global_undo = global_undo

# calc_callback using depsgraph functions
def calc_callback_dg(self, context):
	return calc_callback(self, context, get_inverse_parents_depsgraph, get_matrix_any_depsgraph)

# calc_callback using custom evaluation functions
def calc_callback_ce(self, context):
	return calc_callback(self, context, get_inverse_parents, get_matrix_any_custom_eval)


# draw in 3d-view
def draw_callback(self, context):
	# Remove handler if file was changed and we lose access to self
	try:
		self.properties
	except:
		if global_mtrail_handler_draw:
			bpy.types.SpaceView3D.draw_handler_remove(global_mtrail_handler_draw, 'WINDOW')
		return
	
	mt: MotionTrailProps = context.window_manager.motion_trail

	# polling
	if (context.mode not in ('OBJECT', 'POSE') or not mt.enabled):
		return

	# display limits
	if mt.path_before != 0:
		limit_min = context.scene.frame_current - \
			mt.path_before
	else:
		limit_min = -1e6
	if mt.path_after != 0:
		limit_max = context.scene.frame_current + mt.path_after
	else:
		limit_max = 1e6

	colors_cooked = {}
	chans = [(True, False, False), (False, True, False), (False, False, True), (True, True, False), (False, True, True), (True, False, True), (True, True, True)]
	colors_base = [tuple(mt.handle_color_loc), tuple(mt.handle_color_rot), tuple(mt.handle_color_scl)]
	zeroadd = [0.0, 0.0, 0.0, 0.0]
	zeromul = [1.0, 1.0, 1.0, 1.0]
	for c in chans:
		colors_add = [(colors_base[i] if c[i] else zeroadd) for i in range(3)]
		colors_mul = [(colors_base[i] if c[i] else zeromul) for i in range(3)]

		mulled = mul4(colors_mul[0], mul4(colors_mul[1], colors_mul[2]))
		added = add4(colors_add[0], add4(colors_add[1], colors_add[2]))

		final = lerp4(mt.handle_color_fac, added, mulled)

		colors_cooked[c] = final


	# draw motion path
	width = mt.path_width
	#uniform_line_shader = gpu.shader.from_builtin('3D_POLYLINE_UNIFORM_COLOR')
	colored_line_shader = gpu.shader.from_builtin('3D_POLYLINE_SMOOTH_COLOR')
	colored_points_shader = gpu.shader.from_builtin('2D_FLAT_COLOR')
	
	poss = []
	cols = []
	
	if mt.path_style == 'simple':
		#uniform_line_shader.bind()
		#uniform_line_shader.uniform_float("color", mt.simple_color)
		#uniform_line_shader.uniform_float("lineWidth", width)
		
		colored_line_shader.bind()
		colored_line_shader.uniform_float("lineWidth", width)
		simple_color = mt.simple_color
		for ob, path in self.paths.items():
			for x, y, color, frame in path:
				if frame < limit_min or frame > limit_max:
					continue
				poss.append((x, y, 0))
				cols.append(simple_color)
			#batch = batch_for_shader(uniform_line_shader, 'LINE_STRIP', {"pos": poss})
			if(not (poss == []) and not (cols == [])):
				batch = batch_for_shader(colored_line_shader, 'LINE_STRIP', {"pos": poss, "color": cols})
				batch.draw(colored_line_shader)
				poss.clear()
				cols.clear()
	else:
		colored_line_shader.bind()
		colored_line_shader.uniform_float("lineWidth", width)
		for ob, path in self.paths.items():
			for i, [x, y, color, frame] in enumerate(path):
				if frame < limit_min or frame > limit_max:
					continue
				if i != 0:
					prev_path = path[i - 1]
					halfway = [(x + prev_path[0]) / 2, (y + prev_path[1]) / 2]

					cols.append(color)
					poss.append((int(halfway[0]), int(halfway[1]), 0.0))
					cols.append(color)
					poss.append((x, y, 0.0))
					
				if i != len(path) - 1:
					next_path = path[i + 1]
					halfway = [(x + next_path[0]) / 2, (y + next_path[1]) / 2]

					cols.append(color)
					poss.append((x, y, 0.0))
					cols.append(color)
					poss.append((int(halfway[0]), int(halfway[1]), 0.0))
			if(not (poss == []) and not (cols == [])):
				batch = batch_for_shader(colored_line_shader, 'LINE_STRIP', {"pos": poss, "color": cols})
				batch.draw(colored_line_shader)
				poss.clear()
				cols.clear()

	# Draw rotation spines
	if mt.show_spines:
		colored_line_shader.bind()
		colored_line_shader.uniform_float("lineWidth", 2)
		poss = []
		cols = []
		for frame, locs in self.spines.items():
			if frame < limit_min or frame > limit_max:
				continue
			
			to_use = (mt.pXspines, mt.pYspines, mt.pZspines, mt.nXspines, mt.nYspines, mt.nZspines)
			to_use_colors = (mt.spine_x_color, mt.spine_y_color, mt.spine_z_color, mt.spine_x_color, mt.spine_y_color, mt.spine_z_color)
			for i in range(6):
				if to_use[i]:
					cols.append(to_use_colors[i])
					poss.append((locs[0][0], locs[0][1], 0.0))
					cols.append(to_use_colors[i])
					poss.append((locs[1][i][0], locs[1][i][1], 0.0))
			if(not (cols == []) and not (poss == [])):
				batch = batch_for_shader(colored_line_shader, 'LINES', {"pos": poss, "color": cols})
				batch.draw(colored_line_shader)
				poss.clear()
				cols.clear()

	if self.highlighted_coord:
		colored_points_shader.bind()

		gpu.state.point_size_set(10.0)
		point_poss = [self.highlighted_coord]
		point_cols = [mt.highlight_color]
		batch = batch_for_shader(colored_points_shader, 'POINTS', {"pos": point_poss, "color": point_cols})
		batch.draw(colored_points_shader)
		point_poss.clear()
		point_cols.clear()

	# draw frames
	if mt.frame_display:
		colored_points_shader.bind()
		point_poss = []
		point_cols = []
		for ob, path in self.paths.items():
			for x, y, color, frame in path:
				if frame < limit_min or frame > limit_max:
					continue
				if self.active_frame and ob == self.active_frame[0] \
				and abs(frame - self.active_frame[1]) < 1e-4:
					point_cols.append(mt.selection_color)
					point_poss.append((x, y))
				else:
					point_poss.append((x, y))
					point_cols.append(mt.frame_color)
			if(not (point_poss == []) and not (point_cols == [])):
				batch = batch_for_shader(colored_points_shader, 'POINTS', {"pos": point_poss, "color": point_cols})
				gpu.state.point_size_set(3.0)
				batch.draw(colored_points_shader)
				point_poss.clear()
				point_cols.clear()

	# time beads are shown in speed and timing modes
	if mt.mode in ('speed', 'timing'):
		gpu.state.point_size_set(4.0)
		point_poss = []
		point_cols = []
		for ob, values in self.timebeads.items():
			for frame, [coords, channels] in values.items():
				if frame < limit_min or frame > limit_max:
					continue
				if self.active_timebead and \
				ob == self.active_timebead[0] and \
				abs(frame - self.active_timebead[1]) < 1e-4:
					point_cols.append(mt.selection_color)
					point_poss.append((coords[0], coords[1]))
				else:
					point_cols.append(mt.timebead_color)
					point_poss.append((coords[0], coords[1]))
			if(not (point_poss == []) and not (point_cols == [])):
				batch = batch_for_shader(colored_points_shader, 'POINTS', {"pos": point_poss, "color": point_cols})
				gpu.state.point_size_set(3.0)
				batch.draw(colored_points_shader)
				point_poss.clear()
				point_cols.clear()

	# handles are only shown in value mode
	if mt.mode == 'values':
		colored_line_shader.bind()
		colored_line_shader.uniform_float("lineWidth", 2)
		poss = []
		cols = []

		for chan in range(3): #TODO: Less magic numbers!
			for ob, values in self.handles.items():
				for frame, sides in values[chan].items():
					if frame < limit_min or frame > limit_max:
						continue
					for side, coords in sides.items():
						if self.active_handle and \
						ob == self.active_handle[0] and \
						side == self.active_handle[2] and \
						abs(frame - self.active_handle[1]) < 1e-4:
							cols.append(mt.selection_color_dark)
							poss.append((self.keyframes[ob][frame][0][0],
								self.keyframes[ob][frame][0][1], 0.0))
							cols.append(mt.selection_color_dark)
							poss.append((coords[0], coords[1], 0.0))
							
						else:
							cols.append(mt.handle_line_color)
							poss.append((self.keyframes[ob][frame][0][0],
								self.keyframes[ob][frame][0][1], 0.0))
							cols.append(mt.handle_line_color)
							poss.append((coords[0], coords[1], 0.0))
			if(not (cols == []) and not (poss == [])):
				batch = batch_for_shader(colored_line_shader, 'LINES', {"pos": poss, "color": cols})
				batch.draw(colored_line_shader)
				poss.clear()
				cols.clear() #TODO: Less drawcalls? Not a big performance concern, sadly, compared to dg

		# draw handles
		colored_points_shader.bind()
		gpu.state.point_size_set(4.0)
		point_poss = []
		point_cols = []
		for chan in range(3):
			for ob, values in self.handles.items():
				for frame, sides in values[chan].items():
					if frame < limit_min or frame > limit_max:
						continue
					for side, coords in sides.items():
						if self.active_handle and \
						ob == self.active_handle[0] and \
						side == self.active_handle[2] and \
						abs(frame - self.active_handle[1]) < 1e-4:
							point_poss.append((coords[0], coords[1]))
							point_cols.append(mt.selection_color)
						else:
							point_poss.append((coords[0], coords[1]))
							point_cols.append(colors_cooked[make_chan(chan)])
		if(not (point_poss == []) and not (point_cols == [])):
			batch = batch_for_shader(colored_points_shader, 'POINTS', {"pos": point_poss, "color": point_cols})
			batch.draw(colored_points_shader)
			point_poss.clear()
			point_cols.clear()

	# draw keyframes
	colored_points_shader.bind()
	gpu.state.point_size_set(6.0)
	point_poss = []
	point_cols = []
	for ob, values in self.keyframes.items():
		for frame, [coords, channels] in values.items():
			if frame < limit_min or frame > limit_max:
				continue
			if self.active_keyframe and \
			ob == self.active_keyframe[0] and \
			abs(frame - self.active_keyframe[1]) < 1e-4:
				point_poss.append((coords[0], coords[1]))
				point_cols.append(mt.selection_color)
			else:
				point_poss.append((coords[0], coords[1]))
				point_cols.append(colors_cooked[channels])
	if(not (point_poss == []) and not (point_cols == [])):
		batch = batch_for_shader(colored_points_shader, 'POINTS', {"pos": point_poss, "color": point_cols})
		batch.draw(colored_points_shader)
		point_poss.clear()
		point_cols.clear()

	# draw keyframe-numbers
	if mt.keyframe_numbers:
		blf.size(0, 12, 72)
		blf.color(0, 1.0, 1.0, 0.0, 1.0)
		for ob, values in self.keyframes.items():
			for frame, [coords, channels]  in values.items():
				if frame < limit_min or frame > limit_max:
					continue
				blf.position(0, coords[0] + 3, coords[1] + 3, 0)
				text = str(frame).split(".")
				if len(text) == 1:
					text = text[0]
				elif len(text[1]) == 1 and text[1] == "0":
					text = text[0]
				else:
					text = text[0] + "." + text[1][0]
				if self.active_keyframe and \
				ob == self.active_keyframe[0] and \
				abs(frame - self.active_keyframe[1]) < 1e-4:
					c = mt.selected_text_color
					blf.color(0, * c)
					blf.draw(0, text)
				else:
					c = mt.text_color
					blf.color(0, * c)
					blf.draw(0, text)

	# Draw constraining indicator
	if self.drag:
		constraint_colors = [\
			[[0.0, 0.0, 0.0, 1.0], [1.0, 0.1, 0.1, 1.0]],
			[[0.0, 0.0, 0.0, 1.0], [0.1, 1.0, 0.1, 1.0]],
			[[0.0, 0.0, 0.0, 1.0], [0.1, 0.1, 1.0, 1.0]]]

		constraint_texts = ["X", "Y", "Z"]
		orient_texts = ["(Global)", "(Local)"]

		#TODO: less hardcoded text positions?
		blf.size(0, 12, 130)
		blf.position(0, 10, 40, 0)
		blf.color(0, 0.0, 0.0, 0.0, 1.0)
		blf.draw(0, "Constraints: ")

		blf.size(0, 12, 170)
		for i in range(3):
			blf.position(0, 150 + i*30, 40, 0)
			blf.color(0, *constraint_colors[i][self.constraint_axes[i]])
			blf.draw(0, constraint_texts[i])

		if self.constraint_axes[0] or self.constraint_axes[1] or self.constraint_axes[2]:
			blf.color(0, 0.0, 0.0, 0.0, 1.0)
			blf.size(0, 12, 100)
			blf.position(0, 250, 40, 0)
			blf.draw(0, orient_texts[self.constraint_orientation])

	# restore opengl defaults
	gpu.state.point_size_set(1.0) # TODO: is this the correct value?

# change data based on mouse movement
def drag(self, context, event, inverse_getter):
	self: MotionTrailOperator # VSCode complains that this type hint is undefined if done in the parameters?!
	mt: MotionTrailProps = context.window_manager.motion_trail

	# change 3d-location of keyframe
	if mt.mode == 'values' and self.active_keyframe:
		ob, frame, frame_ori, chans = self.active_keyframe
		mat = inverse_getter(frame, ob, context)
		
		mouse_ori_world = mat @ screen_to_world(context, self.drag_mouse_ori[0],
			self.drag_mouse_ori[1])
		vector = mat @ screen_to_world(context, event.mouse_region_x,
			event.mouse_region_y)
		d = vector - mouse_ori_world

		loc_ori_ws = 0 # TODO: Use uncached getter here?
		loc_ori_ls = mat @ loc_ori_ws
		new_loc = loc_ori_ls + d
		curves = get_curves(ob)

		for chan in range(len(curves)):
			if not chans[chan]:
				continue

			for fcurv in range(len(curves[chan])):
				for kf in curve.keyframe_points:
					if kf.co[0] == frame:
						kf.co[1] = new_loc[i]
						kf.handle_left[1] = self.keyframes_ori[ob][chan][fcurv][frame][1][1] + d[i]
						kf.handle_right[1] = self.keyframes_ori[ob][chan][fcurv][frame][2][1] + d[i]
						break

	# change 3d-location of handle
	elif mt.mode == 'values' and self.active_handle:
		ob, frame, side = self.active_handle
		mat = inverse_getter(frame, ob, context)

		mouse_ori_world = mat @ screen_to_world(context, self.drag_mouse_ori[0],
			self.drag_mouse_ori[1])
		vector = mat @ screen_to_world(context, event.mouse_region_x,
			event.mouse_region_y)
		d = vector - mouse_ori_world
		curves = get_curves(ob)

		for i, curve in enumerate(curves):
			for kf in curve.keyframe_points:
				if kf.co[0] == frame:
					if side == "left":
						# change handle type, if necessary
						# TODO: function for this, makes code unreadable
						if kf.handle_left_type in (
								'AUTO',
								'AUTO_CLAMPED',
								'ANIM_CLAMPED'):
							kf.handle_left_type = 'ALIGNED'
						elif kf.handle_left_type == 'VECTOR':
							kf.handle_left_type = 'FREE'
						# change handle position(s)
						kf.handle_left[1] = handles_ori[ob][frame]["left"][i][1] + d[i]
						if kf.handle_left_type in (
								'ALIGNED',
								'ANIM_CLAMPED',
								'AUTO',
								'AUTO_CLAMPED'):
							dif = (
								abs(handles_ori[ob][frame]["right"][i][0] -
								kf.co[0]) / abs(kf.handle_left[0] -
								kf.co[0])
								) * d[i]
							kf.handle_right[1] = handles_ori[ob][frame]["right"][i][1] - dif
					elif side == "right":
						# change handle type, if necessary
						if kf.handle_right_type in (
								'AUTO',
								'AUTO_CLAMPED',
								'ANIM_CLAMPED'):
							kf.handle_left_type = 'ALIGNED'
							kf.handle_right_type = 'ALIGNED'
						elif kf.handle_right_type == 'VECTOR':
							kf.handle_left_type = 'FREE'
							kf.handle_right_type = 'FREE'
						# change handle position(s)
						kf.handle_right[1] = handles_ori[ob][frame]["right"][i][1] + d[i]
						if kf.handle_right_type in (
								'ALIGNED',
								'ANIM_CLAMPED',
								'AUTO',
								'AUTO_CLAMPED'):
							dif = (
								abs(handles_ori[ob][frame]["left"][i][0] -
								kf.co[0]) / abs(kf.handle_right[0] -
								kf.co[0])
								) * d[i]
							kf.handle_left[1] = handles_ori[ob][frame]["left"][i][1] - dif
					break

	# change position of all keyframes on timeline
	elif mt.mode == 'timing' and active_timebead:
		ob, frame, frame_ori = active_timebead
		curves = get_curves(ob)
		ranges = [val for c in curves for val in c.range()]
		ranges.sort()
		range_min = round(ranges[0])
		range_max = round(ranges[-1])
		range = range_max - range_min
		dx_screen = -(Vector([event.mouse_region_x,
			event.mouse_region_y]) - self.drag_mouse_ori)[0]
		dx_screen = dx_screen / context.region.width * range
		new_frame = frame + dx_screen
		shift_low = max(1e-4, (new_frame - range_min) / (frame - range_min))
		shift_high = max(1e-4, (range_max - new_frame) / (range_max - frame))

		new_mapping = {}
		for i, curve in enumerate(curves):
			for j, kf in enumerate(curve.keyframe_points):
				frame_map = kf.co[0]
				if frame_map < range_min + 1e-4 or \
				frame_map > range_max - 1e-4:
					continue
				frame_ori = False
				for f in keyframes_ori[ob]:
					if abs(f - frame_map) < 1e-4:
						frame_ori = keyframes_ori[ob][f][0]
						value_ori = keyframes_ori[ob][f]
						break
				if not frame_ori:
					continue
				if frame_ori <= frame:
					frame_new = (frame_ori - range_min) * shift_low + \
						range_min
				else:
					frame_new = range_max - (range_max - frame_ori) * \
						shift_high
				frame_new = max(
							range_min + j, min(frame_new, range_max -
							(len(curve.keyframe_points) - j) + 1)
							)
				d_frame = frame_new - frame_ori
				if frame_new not in new_mapping:
					new_mapping[frame_new] = value_ori
				kf.co[0] = frame_new
				kf.handle_left[0] = handles_ori[ob][frame_ori]["left"][i][0] + d_frame
				kf.handle_right[0] = handles_ori[ob][frame_ori]["right"][i][0] + d_frame
		del keyframes_ori[ob]
		keyframes_ori[ob] = {}
		for new_frame, value in new_mapping.items():
			keyframes_ori[ob][new_frame] = value

	# change position of active keyframe on the timeline
	elif mt.mode == 'timing' and active_keyframe:
		ob, frame, frame_ori = active_keyframe
		mat = inverse_getter(frame, ob, context)

		mouse_ori_world = mat @ screen_to_world(context, drag_mouse_ori[0],
			drag_mouse_ori[1])
		vector = mat @ screen_to_world(context, event.mouse_region_x,
			event.mouse_region_y)
		d = vector - mouse_ori_world

		locs_ori = [[f_ori, coords] for f_mapped, [f_ori, coords] in
					keyframes_ori[ob].items()]
		locs_ori.sort()
		direction = 1
		range = False
		for i, [f_ori, coords] in enumerate(locs_ori):
			if abs(frame_ori - f_ori) < 1e-4:
				if i == 0:
					# first keyframe, nothing before it
					direction = -1
					range = [f_ori, locs_ori[i + 1][0]]
				elif i == len(locs_ori) - 1:
					# last keyframe, nothing after it
					range = [locs_ori[i - 1][0], f_ori]
				else:
					current = Vector(coords)
					next = Vector(locs_ori[i + 1][1])
					previous = Vector(locs_ori[i - 1][1])
					angle_to_next = d.angle(next - current, 0)
					angle_to_previous = d.angle(previous - current, 0)
					if angle_to_previous < angle_to_next:
						# mouse movement is in direction of previous keyframe
						direction = -1
					range = [locs_ori[i - 1][0], locs_ori[i + 1][0]]
				break
		direction *= -1  # feels more natural in 3d-view
		if not range:
			# keyframe not found, is impossible, but better safe than sorry
			return(active_keyframe, active_timebead, keyframes_ori)
		# calculate strength of movement
		d_screen = Vector([event.mouse_region_x,
			event.mouse_region_y]) - self.drag_mouse_ori
		if d_screen.length != 0:
			d_screen = d_screen.length / (abs(d_screen[0]) / d_screen.length *
					  context.region.width + abs(d_screen[1]) / d_screen.length *
					  context.region.height)
			d_screen *= direction  # d_screen value ranges from -1.0 to 1.0
		else:
			d_screen = 0.0
		new_frame = d_screen * (range[1] - range[0]) + frame_ori
		max_frame = range[1]
		if max_frame == frame_ori:
			max_frame += 1
		min_frame = range[0]
		if min_frame == frame_ori:
			min_frame -= 1
		new_frame = min(max_frame - 1, max(min_frame + 1, new_frame))
		d_frame = new_frame - frame_ori
		curves = get_curves(ob)

		for i, curve in enumerate(curves):
			for kf in curve.keyframe_points:
				if abs(kf.co[0] - frame) < 1e-4:
					kf.co[0] = new_frame
					kf.handle_left[0] = handles_ori[ob][frame_ori]["left"][i][0] + d_frame
					kf.handle_right[0] = handles_ori[ob][frame_ori]["right"][i][0] + d_frame
					break
		active_keyframe = [ob, new_frame, frame_ori]

	# change position of active timebead on the timeline, thus altering speed
	elif mt.mode == 'speed' and active_timebead:
		ob, frame, frame_ori = active_timebead
		mat = inverse_getter(frame, ob, context)

		mouse_ori_world = mat @ screen_to_world(context, drag_mouse_ori[0],
			drag_mouse_ori[1])
		vector = mat @ screen_to_world(context, event.mouse_region_x,
			event.mouse_region_y)
		d = vector - mouse_ori_world

		# determine direction (to next or previous keyframe)
		curves = get_curves(ob)
		fcx, fcy, fcz = curves
		locx = fcx.evaluate(frame_ori)
		locy = fcy.evaluate(frame_ori)
		locz = fcz.evaluate(frame_ori)
		loc_ori = Vector([locx, locy, locz])  # bonespace
		keyframes = [kf for kf in keyframes_ori[ob]]
		keyframes.append(frame_ori)
		keyframes.sort()
		frame_index = keyframes.index(frame_ori)
		kf_prev = keyframes[frame_index - 1]
		kf_next = keyframes[frame_index + 1]
		vec_prev = (
				(Matrix.Translation(-loc_ori) @ mat) @ \
				Vector(keyframes_ori[ob][kf_prev][1])).normalized()
		vec_next = (
				(Matrix.Translation(-loc_ori) @ mat) @ \
				Vector(keyframes_ori[ob][kf_next][1])).normalized()
		d_normal = d.copy().normalized()
		dist_to_next = (d_normal - vec_next).length
		dist_to_prev = (d_normal - vec_prev).length
		if dist_to_prev < dist_to_next:
			direction = 1
		else:
			direction = -1

		if (kf_next - frame_ori) < (frame_ori - kf_prev):
			kf_bead = kf_next
			side = "left"
		else:
			kf_bead = kf_prev
			side = "right"
		d_frame = d.length * direction * 2  # * 2 to make it more sensitive

		angles = []
		for i, curve in enumerate(curves):
			for kf in curve.keyframe_points:
				if abs(kf.co[0] - kf_bead) < 1e-4:
					if side == "left":
						# left side
						kf.handle_left[0] = min(
											handles_ori[ob][kf_bead]["left"][i][0] +
											d_frame, kf_bead - 1
											)
						angle = Vector([-1, 0]).angle(
											Vector(kf.handle_left) -
											Vector(kf.co), 0
											)
						if angle != 0:
							angles.append(angle)
					else:
						# right side
						kf.handle_right[0] = max(
											handles_ori[ob][kf_bead]["right"][i][0] +
											d_frame, kf_bead + 1
											)
						angle = Vector([1, 0]).angle(
											Vector(kf.handle_right) -
											Vector(kf.co), 0
											)
						if angle != 0:
							angles.append(angle)
					break

		# update frame of active_timebead
		perc = (sum(angles) / len(angles)) / (math.pi / 2)
		perc = max(0.4, min(1, perc * 5))
		if side == "left":
			bead_frame = kf_bead - perc * ((kf_bead - kf_prev - 2) / 2)
		else:
			bead_frame = kf_bead + perc * ((kf_next - kf_bead - 2) / 2)
		active_timebead = [ob, bead_frame, frame_ori]

	return(active_keyframe, active_timebead, keyframes_ori)


# revert changes made by dragging
def cancel_drag(self, context):
	mt: MotionTrailProps = context.window_manager.motion_trail

	# revert change in values of active keyframe and its handles
	if mt.mode == 'values' and active_keyframe:
		objectname, frame, frame_ori, active_ob, child = active_keyframe
		curves = get_curves(active_ob, child)
		for i, curve in enumerate(curves):
			for kf in curve.keyframe_points:
				if kf.co[0] == frame:
					kf.co[1] = mt.keyframe_backup[i]
					kf.handle_left[1] = handles_ori[objectname][frame]["left"][i][1]
					kf.handle_right[1] = handles_ori[objectname][frame]["right"][i][1]
					break
		mt.backed_up_keyframes = False

	# revert change in value of active handle
	elif mt.mode == 'values' and self.active_handle:
		objectname, frame, side, active_ob, child = self.active_handle
		curves = get_curves(active_ob, child)
		for i, curve in enumerate(curves):
			for kf in curve.keyframe_points:
				if kf.co[0] == frame:
					kf.handle_left[1] = handles_ori[objectname][frame]["left"][i][1]
					kf.handle_right[1] = handles_ori[objectname][frame]["right"][i][1]
					break

	# revert position of all keyframes and handles on timeline
	elif mt.mode == 'timing' and active_timebead:
		objectname, frame, frame_ori, active_ob, child = active_timebead
		curves = get_curves(active_ob, child)
		for i, curve in enumerate(curves):
			for kf in curve.keyframe_points:
				for kf_ori, [frame_ori, loc] in keyframes_ori[objectname].\
				items():
					if abs(kf.co[0] - kf_ori) < 1e-4:
						kf.co[0] = frame_ori
						kf.handle_left[0] = handles_ori[objectname][frame_ori]["left"][i][0]
						kf.handle_right[0] = handles_ori[objectname][frame_ori]["right"][i][0]
						break

	# revert position of active keyframe and its handles on the timeline
	elif mt.mode == 'timing' and active_keyframe:
		objectname, frame, frame_ori, active_ob, child = active_keyframe
		curves = get_curves(active_ob, child)
		for i, curve in enumerate(curves):
			for kf in curve.keyframe_points:
				if abs(kf.co[0] - frame) < 1e-4:
					kf.co[0] = keyframes_ori[objectname][frame_ori][0]
					kf.handle_left[0] = handles_ori[objectname][frame_ori]["left"][i][0]
					kf.handle_right[0] = handles_ori[objectname][frame_ori]["right"][i][0]
					break
		active_keyframe = [objectname, frame_ori, frame_ori, active_ob, child]

	# revert position of handles on the timeline
	elif mt.mode == 'speed' and active_timebead:
		objectname, frame, frame_ori, active_ob, child = active_timebead
		curves = get_curves(active_ob, child)
		keyframes = [kf for kf in keyframes_ori[objectname]]
		keyframes.append(frame_ori)
		keyframes.sort()
		frame_index = keyframes.index(frame_ori)
		kf_prev = keyframes[frame_index - 1]
		kf_next = keyframes[frame_index + 1]
		if (kf_next - frame_ori) < (frame_ori - kf_prev):
			kf_frame = kf_next
		else:
			kf_frame = kf_prev
		for i, curve in enumerate(curves):
			for kf in curve.keyframe_points:
				if kf.co[0] == kf_frame:
					kf.handle_left[0] = handles_ori[objectname][kf_frame]["left"][i][0]
					kf.handle_right[0] = handles_ori[objectname][kf_frame]["right"][i][0]
					break
		active_timebead = [objectname, frame_ori, frame_ori, active_ob, child]

	return(active_keyframe, active_timebead)


# return the handle type of the active selection
def get_handle_type(active_keyframe, active_handle):
	if active_keyframe:
		ob, frame, side = active_keyframe
		side = "both"
	elif active_handle:
		ob, frame, side = active_handle
	else:
		# no active handle(s)
		return (False)

	# properties used when changing handle type
	bpy.context.window_manager.motion_trail.handle_type_frame = frame
	bpy.context.window_manager.motion_trail.handle_type_side = side
	bpy.context.window_manager.motion_trail.handle_type_action_ob = ob

	curves = get_curves(ob)
	for c in curves:
		for kf in c.keyframe_points:
			if kf.co[0] == frame:
				if side in ("left", "both"):
					return(kf.handle_left_type)
				else:
					return(kf.handle_right_type)

	return("AUTO")


# Turn the given frame into a keyframe
# TODO: Specifier for channels
def insert_keyframe(self, context, frame):
	ob, frame, frame = frame
	curves = get_curves(ob)
	for c in curves:
		y = c.evaluate(frame)
		if c.keyframe_points:
			c.keyframe_points.insert(frame, y)

	bpy.context.window_manager.motion_trail.force_update = True
	calc_callback(self, context)


# change the handle type of the active selection
def set_handle_type(self, context):

	mt: MotionTrailProps = context.window_manager.motion_trail

	if not mt.handle_type_enabled:
		return
	if mt.handle_type_old == mt.handle_type:
		# function called because of selection change, not change in type
		return
	mt.handle_type_old = mt.handle_type

	frame = mt.handle_type_frame
	side = mt.handle_type_side
	action_ob = mt.handle_type_action_ob
	action_ob = bpy.data.objects[action_ob]
	new_type = mt.handle_type

	curves = get_curves(action_ob)
	for c in curves:
		for kf in c.keyframe_points:
			if kf.co[0] == frame:
				# align if necessary
				if side in ("right", "both") and new_type in (
							"AUTO", "AUTO_CLAMPED", "ALIGNED"):
					# change right handle
					normal = (kf.co - kf.handle_left).normalized()
					size = (kf.handle_right[0] - kf.co[0]) / normal[0]
					normal = normal * size + kf.co
					kf.handle_right[1] = normal[1]
				elif side == "left" and new_type in (
							"AUTO", "AUTO_CLAMPED", "ALIGNED"):
					# change left handle
					normal = (kf.co - kf.handle_right).normalized()
					size = (kf.handle_left[0] - kf.co[0]) / normal[0]
					normal = normal * size + kf.co
					kf.handle_left[1] = normal[1]
				# change type
				if side in ("left", "both"):
					kf.handle_left_type = new_type
				if side in ("right", "both"):
					kf.handle_right_type = new_type

	mt.force_update = True

def force_update_callback(self, context):
	# Remove handler if file was changed and we lose access to self
	try:
		self.properties
	except:
		if global_mtrail_handler_update:
			bpy.types.SpaceGraphEditor.draw_handler_remove(global_mtrail_handler_update, 'WINDOW')
		if global_mtrail_msgbus_owner:
			bpy.msgbus.clear_by_owner(global_mtrail_msgbus_owner)
		
		return
	
	context.window_manager.motion_trail.force_update = True


global_mtrail_handler_calc = None
global_mtrail_handler_draw = None
global_mtrail_handler_update = None
global_mtrail_msgbus_owner = None

class MotionTrailOperator(bpy.types.Operator):
	bl_idname = "view3d.motion_trail"
	bl_label = "Motion Trail"
	bl_description = "Edit motion trails in 3d-view"
	bl_options = {'REGISTER'}

	_handle_calc = None
	_handle_draw = None
	_handle_update = None
	_timer = None

	drag: bool 
	"""Whether or not we're dragging"""
	lock: bool
	"""Whether or not we're changing the motion trail"""

	constraint_axes: list[bool] = [False, False, False]
	"""Bools for which axes are constrained. Please keep 3 long."""
	constraint_orientation: bool = 0
	"""0/False = Global, 1/True = Local"""

	click: dict(Object, list[any])
	"""Items that may be clicked on. Structure: {ob: [[frame, type, coord, channels], ...], ob2: ...}"""

	keyframes_ori: dict(Object, list[list[dict(float, list[list[float]])]])
	"""{ob: [chan: [fcurve: {frame: ([x, y] x3), ...} x 3/4?] x3?], ...},\n
	where the 3 Vectors are the coordinates of the keyframe, its left handle, and its right handle."""

	active_keyframe: list[any]
	"""If a keyframe is active, this contains [ob, frame, frame, channels]""" 
	active_handle: list[any]
	"""If a handle is active, this contains [ob, frame, 'left'/'right', channels]"""
	active_timebead: list[any]
	"""If a timebead is active, this contains [ob, frame, frame, channels]"""
	active_frame: list[any] 
	"""If a frame is active, this contains [ob, frame, frame, channels]"""

	@staticmethod
	def handle_add(self, context):

		if not context.window_manager.motion_trail.use_depsgraph:
			global global_mtrail_handler_calc
			global_mtrail_handler_calc = \
			MotionTrailOperator._handle_calc = bpy.types.SpaceView3D.draw_handler_add(
				calc_callback_ce, (self, context), 'WINDOW', 'POST_VIEW')

			global global_mtrail_handler_update
			global_mtrail_handler_update = \
			MotionTrailOperator._handle_update = bpy.types.SpaceGraphEditor.draw_handler_add(
			force_update_callback, (self, context), 'WINDOW', 'POST_PIXEL')

			global global_mtrail_msgbus_owner
			bpy.msgbus.subscribe_rna(
    			key=(bpy.types.Keyframe, "co_ui"), # why doesn't simply "co" work?
    			owner=global_mtrail_msgbus_owner,
    			args=(self, context),
    			notify=force_update_callback
			)
		
		global global_mtrail_handler_draw
		global_mtrail_handler_draw = \
		MotionTrailOperator._handle_draw = bpy.types.SpaceView3D.draw_handler_add(
			draw_callback, (self, context), 'WINDOW', 'POST_PIXEL')

	@staticmethod
	def handle_remove():
		if MotionTrailOperator._handle_calc is not None:
			try:
				bpy.types.SpaceView3D.draw_handler_remove(MotionTrailOperator._handle_calc, 'WINDOW')
			except:
				pass
		if MotionTrailOperator._handle_draw is not None:
			try:
				bpy.types.SpaceView3D.draw_handler_remove(MotionTrailOperator._handle_draw, 'WINDOW')
			except:
				pass
		if MotionTrailOperator._handle_update is not None:
			try:
				bpy.types.SpaceGraphEditor.draw_handler_remove(MotionTrailOperator._handle_update, 'WINDOW')
			except:
				pass

		if MotionTrailOperator._handle_update is not None:
			try:
				bpy.msgbus.clear_by_owner(global_mtrail_msgbus_owner)
			except:
				pass
			
		MotionTrailOperator._handle_calc = None
		MotionTrailOperator._handle_draw = None
		MotionTrailOperator._handle_update = None

	def modal(self, context, event):
		# XXX Required, or custom transform.translate will break!
		# XXX If one disables and re-enables motion trail, modal op will still be running,
		# XXX default translate op will unintentionally get called, followed by custom translate.

		mt: MotionTrailProps = context.window_manager.motion_trail

		if not mt.enabled:
			MotionTrailOperator.handle_remove()
			if context.area:
				context.area.tag_redraw()
			return {'FINISHED'}

		if mt.use_depsgraph:
			calc_callback_dg(self, context)

		#if not context.area or not context.region: or event.type == 'NONE':
			#context.area.tag_redraw()
		#	return {'PASS_THROUGH'}

		no_passthrough = False

		if event.type in ['X', 'Y', 'Z'] and event.value == 'PRESS':
			no_passthrough = True
			new_constraint = []

			if not event.shift:
				if event.type == 'X':
					new_constraint = [True, False, False]
				elif event.type == 'Y':
					new_constraint = [False, True, False]
				else:
					new_constraint = [False, False, True]
			else:
				if event.type == 'X':
					new_constraint = [False, True, True]
				elif event.type == 'Y':
					new_constraint = [True, False, True]
				else:
					new_constraint = [True, True, False]

			if self.constraint_axes == new_constraint:
				if not self.constraint_orientation:
					self.constraint_orientation = True
				else:
					self.constraint_orientation = False
					new_constraint = [False, False, False]
			else:
				self.constraint_orientation = False
			
			self.constraint_axes = new_constraint

		# TODO: Copypasted code, any better approach? /\

		if (not context.active_object or
				context.active_object.mode not in ('OBJECT', 'POSE')):
			if self.drag:
				self.drag = False
				self.lock = True
				mt.force_update = True
				mt.backed_up_keyframes = False
			# default hotkeys should still work
			if event.type == self.transform_key and event.value == 'PRESS':
				if bpy.ops.transform.translate.poll():
					bpy.ops.transform.translate('INVOKE_DEFAULT')
			return {'PASS_THROUGH'}
		# check if event was generated within 3d-window, dragging is exception
		if not self.drag:
			if not (0 < event.mouse_region_x < context.region.width) or \
			not (0 < event.mouse_region_y < context.region.height):
				return {'PASS_THROUGH'}

		select = mt.select_key
		deselect_nohit = mt.deselect_nohit_key
		deselect_always = mt.deselect_always_key
		if (event.type == self.transform_key and event.value == 'PRESS' and
			   (self.active_keyframe or
				self.active_handle or
				self.active_timebead or
				self.active_frame)):
			# override default translate()
			if not self.drag:
				# start drag
				if self.active_frame:
					insert_keyframe(self, context, self.active_frame)
					self.active_keyframe = self.active_frame
					self.active_frame = False

				self.keyframes_ori = get_original_animation_data(context, self.keyframes)
				self.drag_mouse_ori = Vector([event.mouse_region_x, event.mouse_region_y])
				self.drag = True
				self.lock = False

				self.constraint_axes = [False, False, False]
				self.constraint_orientation = False

			else:
				# stop drag
				self.drag = False
				self.lock = True
				mt.force_update = True
				mt.backed_up_keyframes = False
		elif event.type == self.transform_key and event.value == 'PRESS':
			# call default translate()
			if bpy.ops.transform.translate.poll():
				bpy.ops.transform.translate('INVOKE_DEFAULT')
		elif (event.type == 'ESC' and self.drag and event.value == 'PRESS') or \
			 (event.type == 'RIGHTMOUSE' and self.drag and event.value == 'PRESS'):
			# cancel drag
			self.drag = False
			self.lock = True
			mt.force_update = True
			mt.backed_up_keyframes = False
			self.active_keyframe, self.active_timebead = cancel_drag(context,
				self.active_keyframe, self.active_handle,
				self.active_timebead, self.keyframes_ori, self.handles_ori)
		elif event.type == 'MOUSEMOVE' and self.drag:
			# drag
			self.active_keyframe, self.active_timebead, self.keyframes_ori = \
				drag(context, event, self.drag_mouse_ori,
				self.active_keyframe, self.active_handle,
				self.active_timebead, self.keyframes_ori, self.handles_ori, get_inverse_parents_depsgraph)
			no_passthrough = True

		elif not self.drag and not event.shift and not event.alt and not event.ctrl:
			# Select or highlight
			threshold = mt.select_threshold
			clicked = Vector([event.mouse_region_x, event.mouse_region_y])

			#context.window_manager.motion_trail.force_update = True
			#TODO: Stare at the above line, should it be commented out?

			found = False

			if mt.path_before == 0:
				frame_min = context.scene.frame_start
			else:
				frame_min = max(
							context.scene.frame_start,
							context.scene.frame_current -
							mt.path_before
							)
			if mt.path_after == 0:
				frame_max = context.scene.frame_end
			else:
				frame_max = min(
							context.scene.frame_end,
							context.scene.frame_current +
							mt.path_after
							)

			for ob, values in self.click.items():
				if found:
					break
				for frame, type, coord, channels in values:
					if frame < frame_min or frame > frame_max:
						continue
					if (coord - clicked).length <= threshold:
						found = True

						if event.type == 'MOUSEMOVE':
							self.highlighted_coord = coord

						if event.type == select: # and event.value == 'PRESS'?
							self.active_keyframe = False
							self.active_handle = False
							self.active_timebead = False
							self.active_frame = False
							mt.handle_type_enabled = True
							no_passthrough = True

							if type == "keyframe":
								self.active_keyframe = [ob, frame, frame, channels]
							elif type == "handle_left":
								self.active_handle = [ob, frame, "left", channels]
							elif type == "handle_right":
								self.active_handle = [ob, frame, "right", channels]
							elif type == "timebead":
								self.active_timebead = [ob, frame, frame, channels]
							elif type == "frame":
								self.active_frame = [ob, frame, frame, channels]
							break
			if not found:
				self.highlighted_coord = None
				if event.type == deselect_nohit: # and event.value == 'PRESS'?
					attrs = ["active_keyframe", "active_handle", "active_timebead", "active_frame"]
					# If a change happens, then no passthrough
					gotten = [getattr(self, attr) for attr in attrs]
					no_passthrough = (not reduce(lambda accum, next: accum and not next, gotten, True)) and not mt.deselect_passthrough
					
					for attr in attrs:
						setattr(self, attr, False)
					mt.handle_type_enabled = False
					
				pass
			else:
				handle_type = get_handle_type(self.active_keyframe,
					self.active_handle)
				if handle_type:
					mt.handle_type_old = handle_type
					mt.handle_type = handle_type
				else:
					mt.handle_type_enabled = False
		elif event.type == 'LEFTMOUSE' and event.value == 'PRESS' and \
		self.drag:
			# stop drag
			self.drag = False
			self.lock = True
			mt.force_update = True
			mt.backed_up_keyframes = False
			bpy.ops.ed.undo_push(message="Confirmed Motion Trail drag")
			no_passthrough = True

		elif event.type == deselect_always and event.value == 'PRESS' and \
		not self.drag and not event.shift and not event.alt and not \
		event.ctrl:
			self.active_keyframe = False
			self.active_handle = False
			self.active_timebead = False
			self.active_frame = False
			mt.handle_type_enabled = False

		if context.area:  # not available if other window-type is fullscreen
			context.area.tag_redraw()

		if no_passthrough:
			return {'RUNNING_MODAL'}

		return {'PASS_THROUGH'}

	def invoke(self, context, event):
		if context.area.type != 'VIEW_3D':
			self.report({'WARNING'}, "View3D not found, cannot run operator")
			return {'CANCELLED'}

		# get clashing keymap items
		wm = context.window_manager
		keyconfig = wm.keyconfigs.active
		select = wm.motion_trail.select_key
		kms = [
			bpy.context.window_manager.keyconfigs.active.keymaps['3D View'],
			bpy.context.window_manager.keyconfigs.active.keymaps['Object Mode']
			]
		kmis = []
		for km in kms:
			for kmi in km.keymap_items:
				if kmi.idname == "transform.translate" and \
				kmi.map_type == 'KEYBOARD' and not \
				kmi.properties.texture_space:
					kmis.append(kmi)
					self.transform_key = kmi.type

		mt: MotionTrailProps = context.window_manager.motion_trail

		if not mt.enabled:
			# enable
			self.active_keyframe = False
			self.active_handle = False
			self.active_timebead = False
			self.active_frame = False
			self.click = {}
			self.drag = False
			self.lock = True
			self.perspective = context.region_data.perspective_matrix
			self.displayed = []
			self.paths = {}
			self.keyframes = {}
			self.handles = [{}, {}, {}]
			self.timebeads = {}
			self.spines = {} 
			self.constraint_axes = [False, False, False]
			self.constraint_orientation = False
			self.affect_all_channels = False

			self.highlighted_coord = None
			self.last_frame = -1

			mt.force_update = True
			mt.backed_up_keyframes = False
			mt.handle_type_enabled = False
			getter = get_matrix_any_depsgraph if mt.use_depsgraph else get_matrix_any_custom_eval
			self.cache = MatrixCache(getter)

			for kmi in kmis:
				kmi.active = False

			MotionTrailOperator.handle_add(self, context)
			mt.enabled = True

			if context.area:
				context.area.tag_redraw()

			context.window_manager.modal_handler_add(self)
			bpy.ops.ed.undo_push(message="Started Motion Trail modal operator")

			self._timer = wm.event_timer_add(0.0, window = context.window)
			return {'RUNNING_MODAL'}

		else:
			# disable
			for kmi in kmis:
				kmi.active = True
			MotionTrailOperator.handle_remove()
			mt.enabled = False

			if context.area:
				context.area.tag_redraw()

			return {'FINISHED'}

	def cancel(self, context):
		context.window_manager.event_timer_remove(self._timer)

def load_defaults(context):
	prefs = context.preferences.addons[__name__].preferences
	flat_props = flatten(configurable_props)
	for p in flat_props:
		default = getattr(prefs.default_trail_settings, p)
		setattr(context.window_manager.motion_trail, p, default)

class MotionTrailLoadDefaults(bpy.types.Operator):
	bl_idname="view3d.motion_trail_load_defaults"
	bl_label="Load Defaults"
	bl_description="Reset all the current settings to match what's in the addon's preferences"
	
	def execute(self, context):
		load_defaults(context)
		return {'FINISHED'}

def save_defaults(context):
	prefs = context.preferences.addons[__name__].preferences
	flat_props = flatten(configurable_props)
	for p in flat_props:
		current = getattr(context.window_manager.motion_trail, p)
		setattr(prefs.default_trail_settings, p, current)

class MotionTrailSaveDefaults(bpy.types.Operator):
	bl_idname="view3d.motion_trail_save_defaults"
	bl_label="Save Defaults"
	bl_description="Overwrite the defaults in the addon's preferences with what the current settings are"

	def draw(self, context):
		layout = self.layout
		col = layout.column()
		col.label(text="Are you sure you want to overwrite the defaults?")
		col.label(text="If not, click off of this dialog box, or ESCape.")
	
	def execute(self, context):
		save_defaults(context)
		return {'FINISHED'}

	def invoke(self, context, event):
		wm = context.window_manager
		return wm.invoke_props_dialog(self)

class MotionTrailPanel(bpy.types.Panel):
	bl_idname = "VIEW3D_PT_motion_trail"
	bl_category = "Animation"
	bl_space_type = 'VIEW_3D'
	bl_region_type = 'UI'
	bl_label = "Motion Trail"
	bl_options = {'DEFAULT_CLOSED'}

	@classmethod
	def poll(cls, context):
		if context.active_object is None:
			return False
		return context.active_object.mode in ('OBJECT', 'POSE')

	def draw(self, context):
		mt: MotionTrailProps = context.window_manager.motion_trail

		if (not mt.loaded_defaults):
			load_defaults(context)
			mt.loaded_defaults = True
		
		col = self.layout.column()
		if not mt.enabled:
			col.operator("view3d.motion_trail", text="Enable motion trail")
		else:
			col.operator("view3d.motion_trail", text="Disable motion trail")

		self.layout.column().prop(mt, "use_depsgraph")

		row = self.layout.column().row()
		row.prop(mt, "do_location")
		row.prop(mt, "do_rotation")
		row.prop(mt, "do_scale")

		box = self.layout.box()
		box.prop(mt, "mode")
		# box.prop(mt, "calculate")
		if mt.mode == 'timing':
			box.prop(mt, "timebeads")

		box = self.layout.box()
		col = box.column()
		row = col.row()

		if mt.path_display:
			row.prop(mt, "path_display", icon="DOWNARROW_HLT", text="", emboss=False)
		else:
			row.prop(mt, "path_display", icon="RIGHTARROW", text="", emboss=False)

		row.label(text="Path options")

		if mt.path_display:
			col.prop(mt, "path_style",
				text="Style")
			
			if mt.path_style == 'simple':
				col.row().prop(mt, "simple_color")
			elif mt.path_style == 'speed':
				col.row().prop(mt, "speed_color_min")
				col.row().prop(mt, "speed_color_max")
			else:
				col.row().prop(mt, "accel_color_neg")	
				col.row().prop(mt, "accel_color_static")
				col.row().prop(mt, "accel_color_pos")
				
			grouped = col.column(align=True)
			grouped.prop(mt, "path_width", text="Width")
			grouped.prop(mt, "path_step")
			row = grouped.row(align=True)
			row.prop(mt, "path_before")
			row.prop(mt, "path_after")
			col = col.column(align=True)
			col.prop(mt, "keyframe_numbers")
			if mt.keyframe_numbers:
				col.row().prop(mt, "text_color")
				col.row().prop(mt, "selected_text_color")
			col.prop(mt, "frame_display")
			if mt.frame_display:
				col.row().prop(mt, "frame_color")

		box = self.layout.box()
		col = box.column(align=True)
		if mt.mode == 'values':
			col.prop(mt, "handle_display",
				text="Handles")
			if mt.handle_display:
				row = col.row()
				row.enabled = mt.handle_type_enabled
				row.prop(mt, "handle_type")
				col.prop(mt, "handle_direction")
				col.prop(mt, "handle_length")
				
				col.row().prop(mt, "handle_line_color")
				col.row().prop(mt, "selection_color_dark")
		else:
			col.row().prop(mt, "timebead_color")

		box = self.layout.box()
		col = box.column(align=True)
		col.row().label(text="Handle colors:")
		handle_color_row = col.row()
		handle_color_row.prop(mt, "handle_color_loc", text="Loc")
		handle_color_row.prop(mt, "handle_color_rot", text="Rot")
		handle_color_row.prop(mt, "handle_color_scl", text="Scale")

		handle_fac_row = col.row()
		handle_fac_row.prop(mt, "handle_color_fac")
		#handle_fac_row.prop(mt, "handle_color_mul", "Mul fac")
		col.row().prop(mt, "selection_color")
		col.row().prop(mt, "highlight_color")
		col.row().prop(mt, "select_key")
		col.row().prop(mt, "select_threshold")
		col.row().prop(mt, "deselect_nohit_key")
		col.row().prop(mt, "deselect_always_key")
		col.row().prop(mt, "deselect_passthrough")
		col.label(text="For the time being, confirm/cancel")
		col.label(text="is LMB/RMB or Esc")

		# Spines
		box = self.layout.box()
		col = box.column(align=True)
		col.row().prop(mt, "show_spines")
		if mt.show_spines:
			pSpineStrings = ["pXspines", "pYspines", "pZspines"]
			nSpineStrings = ["nXspines", "nYspines", "nZspines"]
			spineColorStrings = ["spine_x_color", "spine_y_color", "spine_z_color"]
			
			spine_do_row = col.row()
			spine_do_row.prop(mt, "spine_do_rotation")
			spine_do_row.prop(mt, "spine_do_scale")

			col.row().prop(mt, "spine_step")
			col.row().prop(mt, "spine_length")
			col.row().prop(mt, "spine_offset")

			row = col.row()
			for s in pSpineStrings:
				row.prop(mt, s)

			row = col.row()
			for s in nSpineStrings:
				row.prop(mt, s)

			row = col.row()
			for s in spineColorStrings:
				row.prop(mt, s)
		
		self.layout.column().operator("view3d.motion_trail_load_defaults")
		self.layout.column().operator("view3d.motion_trail_save_defaults")

DESELECT_WARNING = "Deselection will happen before your click registers to the rest of Blender.\n" +\
	"This can prevent you from changing the handle type if it's set to left click"

class MotionTrailProps(bpy.types.PropertyGroup):
	def internal_update(self, context):
		context.window_manager.motion_trail.force_update = True
		if context.area:
			context.area.tag_redraw()

	# internal use
	enabled: BoolProperty(default=False)
	
	loaded_defaults: BoolProperty(default=False)

	force_update: BoolProperty(name="internal use",
		description="Force calc_callback to fully execute",
		default=False)
		
	backed_up_keyframes: BoolProperty(default=False)

	handle_type_enabled: BoolProperty(default=False)
	handle_type_frame: FloatProperty()
	handle_type_side: StringProperty()
	handle_type_action_ob: StringProperty()

	handle_type_old: EnumProperty(
			items=(
				("AUTO", "", ""),
				("AUTO_CLAMPED", "", ""),
				("VECTOR", "", ""),
				("ALIGNED", "", ""),
				("FREE", "", "")),
			default='AUTO'
			)
	# visible in user interface
	calculate: EnumProperty(name="Calculate", items=(
			("fast", "Fast", "Recommended setting, change if the "
							 "motion path is positioned incorrectly"),
			("full", "Full", "Takes parenting and modifiers into account, "
							 "but can be very slow on complicated scenes")),
			description="Calculation method for determining locations",
			default='full',
			update=internal_update
			)
	frame_display: BoolProperty(name="Frames",
			description="Display individual frames as manipulateable dots.\nClick and drag on one to make a new keyframe",
			default=True,
			update=internal_update
			)
	handle_display: BoolProperty(name="Display",
			description="Display keyframe handles",
			default=True,
			update=internal_update
			)
	handle_type: EnumProperty(name="Type", items=(
			("AUTO", "Automatic", ""),
			("AUTO_CLAMPED", "Auto Clamped", ""),
			("VECTOR", "Vector", ""),
			("ALIGNED", "Aligned", ""),
			("FREE", "Free", "")),
			description="Set handle type for the selected handle",
			default='AUTO',
			update=set_handle_type
			)
	keyframe_numbers: BoolProperty(name="Keyframe numbers",
			description="Display keyframe numbers",
			default=False,
			update=internal_update
			)
	mode: EnumProperty(name="Mode", items=(
			("values", "Values", "Alter values of the keyframes"),
			("speed", "Speed", "Change speed between keyframes"),
			("timing", "Timing", "Change position of keyframes on timeline")),
			description="Enable editing of certain properties in the 3d-view",
			default='values',
			update=internal_update
			)
	path_after: IntProperty(name="After",
			description="Number of frames to show after the current frame, "
						"0 = display all",
			default=50,
			min=0,
			update=internal_update
			)
	path_before: IntProperty(name="Before",
			description="Number of frames to show before the current frame, "
						"0 = display all",
			default=50,
			min=0,
			update=internal_update
			)
	path_display: BoolProperty(name="Path options",
			description="Display path options",
			default=True
			)
	path_step: IntProperty(name="Step",
			description="Step size for the frames the motion trail consists of\nIncrease to improve performance",
			default=1,
			min=1,
			max=10,
			update=internal_update
			)
	path_style: EnumProperty(name="Path style", items=(
			("acceleration", "Acceleration", "Gradient based on relative acceleration"),
			("simple", "Simple", "Black line"),
			("speed", "Speed", "Gradient based on relative speed")),
			description="Information conveyed by path color",
			default='simple',
			update=internal_update
			)
	path_width: IntProperty(name="Path width",
			description="Width in pixels",
			default=1,
			min=1,
			soft_max=5,
			update=internal_update
			)
	timebeads: IntProperty(name="Time beads",
			description="Number of time beads to display per segment",
			default=5,
			min=1,
			soft_max=10,
			update=internal_update
			)
	handle_length: FloatProperty(name="Handle length",
			description="Handle length multiplier",
			default = 1.0,
			step = 0.15,
			update=internal_update
			)
	handle_direction: EnumProperty(name="Handle direction",
			description="Affect location, euler rotation and scale only, do NOT affect quaternion rotation",
			items=(
			("time", "Time", "Use only the time coordinate of the handles"),
			("wtime", "Weighted Time", "0.75*time + 0.25*location"),
			("value", "Value", "Use only the value coordinate of the handles"),
			("wloc", "Weighted Location", "0.25*time + 0.75*location"),
			("len", "Directional length", "Use the length of the handle, positive for right and negative for left")),
			default='wtime'
			)

	do_location: BoolProperty(name="Location",
			description="Show and work with location keyframes",
			default=True,
			update=internal_update
			)
	do_rotation: BoolProperty(name="Rotation",
			description="Show and work with rotation keyframes",
			default=False,
			update=internal_update
			)
	do_scale: BoolProperty(name="Scale",
			description="Show and work with scale keyframes",
			default=False,
			update=internal_update
			)	
	
			
	#Key stuff
	select_key: EnumProperty(name="Selection key",
			description="Pressing this key will only either select something if nothing is selected, or override an existing selection",
			items=(
			("LEFTMOUSE", "Left Mouse Button", ""),
			("RIGHTMOUSE", "Right Mouse Button", ""),),
			default='LEFTMOUSE'
			)
	deselect_nohit_key: EnumProperty(name="Deselect miss key",
			description="When your mouse is not over a selectable thing, " +\
				"pressing this key will deselect.\n" + DESELECT_WARNING,
			items=(
			("LEFTMOUSE", "Left Mouse Button", ""),
			("RIGHTMOUSE", "Right Mouse Button", ""),
			("NONE", "None", ""),),
			default='RIGHTMOUSE'
			)
	deselect_always_key: EnumProperty(name="Deselect always key",
			description="Pressing this key will always deselect\n" + DESELECT_WARNING,
			items=(
			("LEFTMOUSE", "Left Mouse Button", ""),
			("RIGHTMOUSE", "Right Mouse Button", ""),
			("NONE", "None", ""),),
			default="NONE"
			)
	select_threshold: FloatProperty(name="Selection distance",
			description="Distance in pixels for selecting something",
			default=10.0,
			step=2,
			min=0.0
			)
	deselect_passthrough: BoolProperty(name="Deselect passthrough",
			description="When something in the motion trail is deselected, whether to pass that button press to the rest of Blender or not",
			default=True
			)

	# Spines

	SPSTRSTR = "Show spines for the " # haha it says str str!!! But really, it would be pointless if the name was longer than the string itself
	SPSTREND = " axis. This visualization works for quaternions as well"
	# Using a Bool vector looks very silly in the UI.

	show_spines: BoolProperty(name="Spines",
			description="Show spines for visualizing rotation along the motion trail",
			default=False,
			update=internal_update,
			)


	pXspines: BoolProperty(name="+X",
			description=SPSTRSTR + "+X" + SPSTREND,
			default=False
			)
	pYspines: BoolProperty(name="+Y",
			description=SPSTRSTR + "+Y" + SPSTREND,
			default=True
			)
	pZspines: BoolProperty(name="+Z",
			description=SPSTRSTR + "+Z" + SPSTREND,
			default=False
			)

	nXspines: BoolProperty(name="-X",
			description=SPSTRSTR + "-X" + SPSTREND,
			default=False
			)
	nYspines: BoolProperty(name="-Y",
			description=SPSTRSTR + "-Y" + SPSTREND,
			default=False
			)
	nZspines: BoolProperty(name="-Z",
			description=SPSTRSTR + "-Z" + SPSTREND,
			default=False
			)

	spine_offset: FloatVectorProperty(name="Offset",
			description="Apply this euler rotation to the motion trail rotation to adjust where spines are",
			default=(0.0, 0.0, 0.0),
			size=3,
			subtype='EULER',
			update=internal_update,
			)
	
	spine_length: FloatProperty(name="Length",
			description="How long spines should be",
			default=1.0,
			update=internal_update,
			)

	spine_step: IntProperty(name="Step",
			description="How many frames to step across for each spine, higher = less spines",
			default=1,
			min=1,
			soft_max=10,
			update=internal_update,
			)

	spine_do_rotation: BoolProperty(name="Do Rotation",
			description="Whether the spines will rotate themselves with the object's rotation",
			default=True,
			update=internal_update
			)
	spine_do_scale: BoolProperty(name="Do Scale",
			description="Whether the spines will scale their length with the object's scale",
			default=False,
			update=internal_update
			)

	#Colors
	simple_color: FloatVectorProperty(name="Color",
			description="Color when using simple drawing mode",
			default=(0.0, 0.0, 0.0, 1.0),
			min=0.0, soft_max=1.0,
			size=4,
			subtype='COLOR'
			)
	speed_color_min: FloatVectorProperty(name="Min color",
			description="Color that lowest speed will be colored in",
			default=(0.0, 0.0, 1.0, 1.0),
			min=0.0, soft_max=1.0,
			size=4,
			subtype='COLOR'
			)
	speed_color_max: FloatVectorProperty(name="Max color",
			description="Color that highest speed will be colored in",
			default=(1.0, 0.0, 0.0, 1.0),
			min=0.0, soft_max=1.0,
			size=4,
			subtype='COLOR'
			)
	accel_color_neg: FloatVectorProperty(name="Negative color",
			description="Color that lowest (negative) acceleration will be colored in",
			default=(0.0, 1.0, 0.0, 1.0),
			min=0.0, soft_max=1.0,
			size=4,
			subtype='COLOR'
			)
	accel_color_static: FloatVectorProperty(name="Static color",
			description="Color that 0 acceleration will be colored in",
			default=(1.0, 1.0, 0.0, 1.0),
			min=0.0, soft_max=1.0,
			size=4,
			subtype='COLOR'
			)
	accel_color_pos: FloatVectorProperty(name="Positive color",
			description="Color that highest acceleration will be colored in",
			default=(1.0, 0.0, 0.0, 1.0),
			min=0.0, soft_max=1.0,
			size=4,
			subtype='COLOR'
			)
	keyframe_color: FloatVectorProperty(name="Keyframe color",
			description="Color that unselected keyframes will be colored in",
			default=(1.0, 0.0, 0.0, 1.0),
			min=0.0, soft_max=1.0,
			size=4,
			subtype='COLOR'
			)
	selection_color: FloatVectorProperty(name="Selection color",
			description="Color that selected frames, keyframes and timebeads will be colored in",
			default=(1.0, 0.5, 0.0, 1.0),
			min=0.0, soft_max=1.0,
			size=4,
			subtype='COLOR'
			)
	selection_color_dark: FloatVectorProperty(name="Handle selection color",
			description="Color that selected handles will be colored in",
			default=(0.75, 0.25, 0.0, 1.0),
			min=0.0, soft_max=1.0,
			size=4,
			subtype='COLOR'
			)

	handle_color_loc: FloatVectorProperty(name="Location handle color",
			description="Color that unselected location handles and keyframes will be colored in",
			default=(1.0, 0.1, 0.1, 1.0),
			min=0.0, soft_max=1.0,
			size=4,
			subtype='COLOR'
			)
	handle_color_rot: FloatVectorProperty(name="Rotation handle color",
			description="Color that unselected rotation handles and keyframes will be colored in",
			default=(0.1, 1.0, 0.1, 1.0),
			min=0.0, soft_max=1.0,
			size=4,
			subtype='COLOR'
			)
	handle_color_scl: FloatVectorProperty(name="Scale handle color",
			description="Color that unselected scale handles and keyframes will be colored in",
			default=(0.1, 0.1, 1.0, 1.0),
			min=0.0, soft_max=1.0,
			size=4,
			subtype='COLOR'
			)
	handle_color_fac: FloatProperty(name="Color factor",
			description="Factor for how much of the added or multiplied color is included in the final handle color",
			default=1.0,
			soft_min=0.0, soft_max=1.0,
			)

	handle_line_color: FloatVectorProperty(name="Handle line color",
			description="Color that unselected handle lines will be colored in",
			default=(0.0, 0.0, 0.0, 1.0),
			min=0.0, soft_max=1.0,
			size=4,
			subtype='COLOR'
			)
	frame_color: FloatVectorProperty(name="Frame color",
			description="Color that unselected frames will be colored in",
			default=(1.0, 1.0, 1.0, 1.0),
			min=0.0, soft_max=1.0,
			size=4,
			subtype='COLOR'
			)
	timebead_color: FloatVectorProperty(name="Timebead color",
			description="Color that timebeads (in speed/timing mode) will be colored in",
			default=(0.0, 1.0, 0.0, 1.0),
			min=0.0, soft_max=1.0,
			size=4,
			subtype='COLOR'
			)
	text_color: FloatVectorProperty(name="Text color",
			description="Color that unselected frame numbers will be colored in",
			default=(1.0, 1.0, 1.0, 1.0),
			min=0.0, soft_max=1.0,
			size=4,
			subtype='COLOR'
			)
	selected_text_color: FloatVectorProperty(name="Selected text color",
			description="Color that selected frame numbers will be colored in",
			default=(1.0, 1.0, 0.5, 1.0),
			min=0.0, soft_max=1.0,
			size=4,
			subtype='COLOR'
			)

	spine_x_color: FloatVectorProperty(name="X color",
			description="Color that spines corresponding to X rotation will be colored in",
			default=(0.2, 0.0, 0.0, 1.0),
			min=0.0, soft_max=1.0,
			size=4,
			subtype='COLOR'
			)
	spine_y_color: FloatVectorProperty(name="Y color",
			description="Color that spines corresponding to Y rotation will be colored in",
			default=(0.0, 0.2, 0.0, 1.0),
			min=0.0, soft_max=1.0,
			size=4,
			subtype='COLOR'
			)
	spine_z_color: FloatVectorProperty(name="Z color",
			description="Color that spines corresponding to Z rotation will be colored in",
			default=(0.0, 0.0, 0.2, 1.0),
			min=0.0, soft_max=1.0,
			size=4,
			subtype='COLOR'
			)

	highlight_color: FloatVectorProperty(name="Highlight color",
			description="Color that something you're about to select will be highlighted in",
			default=(0.0, 1.0, 0.7, 1.0),
			min=0.0, soft_max=1.0,
			size=4,
			subtype='COLOR'
			)

	use_depsgraph: BoolProperty(name="Use depsgraph",
			description="Whether to use the depsgraph or not.\nChanging this takes effect only when motion trails are not active.\n\nUsing the depsgraph currently has the following ups and downs:\n+ Completely accurate motion trails that factor in all constraints, drivers, and so on.\n- Constantly resets un-keyframed changes to objects with keyframes.\n- Does not update with the graph editor or others.\n- Less performant",
			default=False
			)

	# !! From here on till the end, this code is structured to be easily deletable; Look for another chunk of deletable code shortly below!

	master_version: bpy.props.IntVectorProperty(
		default=(-1, -1, -1)
		)

	experimental_version: bpy.props.IntVectorProperty(
		default=(-1, -1, -1)
		)

	version_checked: bpy.props.BoolVectorProperty(
		default=(False, False),
		size=2
		)


import urllib.request
import re
SOURCE_URL = "https://raw.githubusercontent.com/a-One-Fan/Blender-Motion-Trail-Update/one/src/animation_motion_trail_updated.py"
SOURCE_URL_EXPERIMENTAL = "https://raw.githubusercontent.com/a-One-Fan/Blender-Motion-Trail-Update/other_one/src/animation_motion_trail_updated.py"

def get_version(link):
	response = urllib.request.urlopen(link)

	gpl_done = False
	while not gpl_done:
		line = response.readline()
		
		if response.isclosed():
			return -1
		
		if line.find(b"END GPL") > -1:
			gpl_done = True

	gotten_version = None
	while gotten_version == None:
		line = response.readline()

		if response.isclosed():
			return -2
		
		if line.find(b"version") > -1:
			match = re.search(r"\(([0-9]+),\s*([0-9]+),\s*([0-9]+)\)", line.decode())
			if not match:
				return -3
			gotten_version = (int(match[1]), int(match[2]), int(match[3]))

	response.close()

	return gotten_version

class MotionTrailCheckUpdate(bpy.types.Operator):
	bl_idname="info.motion_trail_check_update"
	bl_label="Check available versions"
	bl_description="Check the versions of the motion trail addon available on github"
	
	def execute(self, context):
		mt: MotionTrailProps = context.window_manager.motion_trail

		mt.version_checked = (False, False)

		version_regular = get_version(SOURCE_URL)
		version_experimental = get_version(SOURCE_URL_EXPERIMENTAL)

		if type(version_regular) is not int:
			mt.master_version = version_regular
			mt.version_checked[0] = True

		if type(version_experimental) is not int:
			mt.experimental_version = version_experimental
			mt.version_checked[1] = True

		return {'FINISHED'}

# returns tup1 < tup2
def compare_ver(tup1, tup2):
	for i in range(len(tup1)):
		if tup1[i]<tup2[i]:
			return True
		if tup1[i]>tup2[i]:
			return False
	return False

# == END of deleteable code ==
			
configurable_props = ["use_depsgraph", "select_key", "select_threshold", "deselect_nohit_key", "deselect_always_key", "deselect_passthrough", "mode", "path_style", 
"simple_color", "speed_color_min", "speed_color_max", "accel_color_neg", "accel_color_static", "accel_color_pos",
"keyframe_color", "frame_color", "selection_color", "selection_color_dark", "highlight_color", 
["handle_color_loc", "handle_color_rot", "handle_color_scl"], "handle_color_fac", "handle_line_color", "timebead_color", 
"text_color", "selected_text_color", "path_width", "path_step", "path_before", "path_after",
"keyframe_numbers", "frame_display", "handle_display", "handle_length", "handle_direction", "show_spines", "spine_length", "spine_step", "spine_offset",
["pXspines", "pYspines", "pZspines"], ["nXspines", "nYspines", "nZspines"], ["spine_x_color", "spine_y_color", "spine_z_color"]]
			
class MotionTrailPreferences(bpy.types.AddonPreferences):
	bl_idname = __name__
	
	default_trail_settings: PointerProperty(type=MotionTrailProps)
	
	def draw(self, context):
		layout = self.layout
		col = layout.column()

		mt: MotionTrailProps = context.window_manager.motion_trail

		# !! Deletable code part 2
		col.operator("info.motion_trail_check_update")
		if mt.version_checked[0] or mt.version_checked[1]:
			if mt.version_checked[0]:
				col.row().label(text="Current master version: {}.{}.{}".format(*mt.master_version))
				if compare_ver(bl_info["version"], mt.master_version):
					col.row().label(text="Please update!")
			if mt.version_checked[1]:
				col.row().label(text="Current experimental version: {}.{}.{}".format(*mt.experimental_version))
		else:
			col.row().label(text="Version not checked yet...")
		#end of deletable code

		col.label(text=DESELECT_WARNING)
		col.label(text="Default values for all settings:")
		col.label(text="")
		for p in configurable_props:
			if type(p) is list:
				row = col.row()
				for subp in p:
					row.prop(self.default_trail_settings, subp)
			else:
				col.row().prop(self.default_trail_settings, p)

classes = (
		MotionTrailProps,
		MotionTrailOperator,
		MotionTrailPanel,
		MotionTrailPreferences,
		MotionTrailLoadDefaults,
		MotionTrailSaveDefaults,
		MotionTrailCheckUpdate,
		)


def register():
	for cls in classes:
		bpy.utils.register_class(cls)

	bpy.types.WindowManager.motion_trail = PointerProperty(
												type=MotionTrailProps
												)


def unregister():
	MotionTrailOperator.handle_remove()
	for cls in classes:
		bpy.utils.unregister_class(cls)

	del bpy.types.WindowManager.motion_trail


if __name__ == "__main__":
	register()
