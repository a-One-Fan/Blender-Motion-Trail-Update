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
	"version": (1, 0, 0),
	"blender": (3, 2, 0),
	"location": "View3D > Toolbar > Motion Trail tab",
	"warning": "Please keep the depsgraph toggle in mind, and remember to save often.",
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

def get_chani(chans):
	i = 0
	while i < len(chans):
		if chans[i]:
			return i
		i += 1
	if i == len(chans):
		return -1

def findlist(elem, arr):
	"""Returns index of elem in arr, -1 if not found"""
	for i in range(len(arr)):
		if arr[i] == elem:
			return i
	
	return -1

def flip_chan(filter, curr, id):
	if not filter[id]:
		return curr
	else:
		return tuple([not curr[i] if i == id else curr[i] for i in range(len(curr))])

def single_chan(filter, id):
	if filter[id]:
		return make_chan(id)
	else:
		i = 0
		while not filter[i]:
			i += 1
		return make_chan(i)

# Flattens recursively.
def flatten(deeplist):
	flatlist = []
	for elem in deeplist:
		if type(elem) is list:
			flatlist.extend(flatten(elem))
		else:
			flatlist.append(elem)
	return flatlist

def flrange(start, stop, step):
	res = []
	i = start
	while i < stop:
		res.append[i]
		i += step

	return res

def vecabs(vec):
	return Vector([abs(val) for val in vec])

def mouse_in_region(event, context):
	return event.mouse_region_x > 0 and event.mouse_region_x < context.region.width and \
				event.mouse_region_y > 0 and event.mouse_region_y < context.region.height

def maprange(oldmin, oldmax, newmin, newmax, val):
	fac = (val - oldmin) / (oldmax - oldmin)
	return newmin * (1.0 - fac) + newmax * fac

def clamp(_min, _max, val):
	if val < _min:
		return _min
	if val > _max:
		return _max
	return val

def lerp(v1, v2, fac):
	return v1*(1.0-fac) + v2*fac

def chop_line(p1: Vector, p2: Vector, amt1, amt2):
	totlen = (p2-p1).length
	if totlen==0:
		return (p1, p2)
	fac1 = amt1/totlen
	fac2 = (totlen-amt2)/totlen
	return (lerp(p1, p2, fac1), lerp(p1, p2, fac2))
	
def angle_bisector(central_point: Vector, p1: Vector, p2: Vector): # Lots of normalizing, slow?
	v1 = central_point - p1
	v2 = central_point - p2
	v1.normalize()
	v2.normalize()
	summed = v1 + v2
	if summed.length < 0.001:
		return Vector((v1.y, -v1.x))

	summed.normalize()
	
	return summed

def line_to_tris(points: list[Vector], colors: list, width: float, outline: float, to_append: tuple[list, list, list]):
	"""Convert a line strip into a list of triangles, with a supplementary list for how far each point is from the centre"""
	if len(points) < 2:
		return

	res_tris = []
	res_dist = []
	res_cols = []
	width = width * 0.5 + 0.5
	bisectors = [None for i in range(len(points))]
	normDir = (points[0] - points[1]).normalized()
	bisectors[0] = Vector((normDir.y, -normDir.x))
	last = len(points)-1
	normDir = (points[last] - points[last-1]).normalized()
	bisectors[last] = Vector((normDir.y, -normDir.x))

	for i in range(1, last):
		bisectors[i] = angle_bisector(points[i], points[i-1], points[i+1])
	
	for i in range(0, last):

		# Bisector direction correction, so that the triangles don't fold over themselves
		if (bisectors[i] - bisectors[i+1]).length < 1.0:
			dirs = [(-1, -1), (1, 1)]
		else:
			dirs = [(-1, 1), (1, -1)]

		for (dir, dir2) in dirs:
			to_append[0].append(points[i])
			to_append[0].append(points[i] + bisectors[i]*(width+outline)*dir2)
			to_append[0].append(points[i+1])

			to_append[1].append(colors[i])
			to_append[1].append(colors[i])
			to_append[1].append(colors[i+1])

			to_append[2].append((0, width, outline))
			to_append[2].append((width+outline, width, outline))
			to_append[2].append((0, width, outline))

			to_append[0].append(points[i+1])
			to_append[0].append(points[i] + bisectors[i]*(width+outline)*dir2)
			to_append[0].append(points[i+1] + bisectors[i+1]*(width+outline)*dir)

			to_append[1].append(colors[i+1])
			to_append[1].append(colors[i])
			to_append[1].append(colors[i+1])

			to_append[2].append((0, width, outline))
			to_append[2].append((width+outline, width, outline))
			to_append[2].append((width+outline, width, outline))

	return

def line_strip_to_lines(points: list[Vector], colors: list, width: float, outline: float, to_append: list[list]):
	if len(points) < 2:
		return
	p2 = []
	c2 = []
	p2.append(points[0])
	c2.append(colors[0])
	for i in range(1, len(points)-1):
		p2.append(points[i])
		p2.append(points[i])
		c2.append(colors[i])
		c2.append(colors[i])
	p2.append(points[len(points)-1])
	c2.append(colors[len(points)-1])

	to_append[0].extend(p2)
	to_append[1].extend(c2)
	
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

	def clear(self):
		self.__mats = {}

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

# Returns the index of the first True 
def logsearch_basic(arr: list[bool]):
	if len(arr) == 0:
		return 0

	hi = len(arr)
	lo = 0

	while hi > lo:
		mid = lo + int( (hi - lo)/2.0 )
		if arr[mid]:
			hi = mid
		else:
			lo = mid + 1
	
	return lo

def logsearch_func(arr, compare):
	return logsearch_basic(list(map(compare, arr)))

class FloatMap():
	__eps: float
	__kvps: list[tuple[float, any]]

	def __init__(self, eps = 0.0001):
		self.__eps = eps
		self.__kvps = []

	def __findi(self, key):
		return logsearch_func(self.__kvps, (lambda tup: (tup[0] + self.__eps) > key))

	def __getitem__(self, key):
		i = self.__findi(key)
		if i >= len(self.__kvps):
			raise KeyError(key)
		
		realkey = self.__kvps[i][0]
		if not (abs(key - realkey) < self.__eps):
			raise KeyError(key, self.__kvps[i][0])
		return self.__kvps[i][1]

	def exists(self, key):
		i = self.__findi(key)

		return (i < len(self.__kvps)) and (abs(key - self.__kvps[i][0]) < self.__eps)

	def items(self):
		return self.__kvps

	def __setitem__(self, key, newval):
		i = self.__findi(key)
		
		if ((i < len(self.__kvps)) and (abs(key - self.__kvps[i][0]) < self.__eps)):
			self.__kvps[i] = (key, newval)
		else:
			self.__kvps.insert(i, (key, newval))

	def __bool__(self):
		return len(self.__kvps) > 0

	def __str__(self):
		return str(self.__kvps)
		

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

	if obj is None or not animDataContainer.animation_data or not animDataContainer.animation_data.action:
		return []
	
	return get_curves_action(obj, animDataContainer.animation_data.action)

# turn screen coordinates (x,y) into world coordinates vector
def screen_to_worldxy(context, x, y):
	depth_vector = view3d_utils.region_2d_to_vector_3d(
							context.region, context.region_data, [x, y]
							)
	vector = view3d_utils.region_2d_to_location_3d(
							context.region, context.region_data, [x, y],
							depth_vector
							)

	return (vector)

def screen_to_world(context, vec):
	return screen_to_worldxy(context, vec.x, vec.y)

# turn 3d world coordinates vector into screen coordinate floats (x,y)
def world_to_screen(context, vector):
	prj = context.region_data.perspective_matrix @ \
		Vector((vector[0], vector[1], vector[2], 1.0))
	width_half = context.region.width / 2.0
	height_half = context.region.height / 2.0

	x = width_half + width_half * (prj.x / prj.w)
	y = height_half + height_half * (prj.y / prj.w)

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
		if not reduce((lambda a, b: a and b), bools, True): # TODO: functional programming unnecessary
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
	return get_matrix_any_custom_eval(frame, ob, False).inverted()

def get_inverse_parents_depsgraph(frame, ob, context):
	mat = ''
	if type(ob) is PoseBone:
		mat = get_matrix_frame(ob, frame, ob.id_data.animation_data.action)
	else:
		mat = get_matrix_frame(ob, frame, ob.animation_data.action)
		
	return (get_matrix_any_depsgraph(frame, ob, context) @ mat.inverted()).inverted()

def get_original_animation_data(context):
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
				keyframes_ori[ob][chan].append(FloatMap())
				kf: Keyframe
				for kf in curves[chan][fcurv].keyframe_points:
					keyframes_ori[ob][chan][fcurv][kf.co[0]] = \
						(kf.co.copy(), kf.handle_left.copy(), kf.handle_right.copy(), "" +  kf.handle_left_type, "" + kf.handle_right_type)

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
			indices2 = enum2[j][1][1]
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
def calc_callback(self, context):
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

	if not self.drag and not selection_change and \
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

	if self.perspective != context.region_data.perspective_matrix:
		self.highlighted_coord = False # TODO: Highlighted coord persists after its highlighted selectable vanish while playing animation... think of fix?
	
	if selection_change or self.drag or mt.force_update:
		self.cache.clear()
		self.cache_inverse.clear()

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
			step = mt.path_step if not self.drag else mt.path_step_drag

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
						mat = self.cache_inverse.get_matrix(frame, ob, context)
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
					click.append( [frame, "timebead", Vector([x, y]), tuple(channels)] )
				self.timebeads[ob] = timebeads

			if mt.show_spines:

				spine_step = max(mt.spine_step, step)
				self.spines[ob] = {}

				for frame in range(range_min, range_max + 1, spine_step):
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
					
					self.spines[ob][frame] = (baseLoc, resLocs)

			# add frame positions to click-list
			if mt.frame_display:
				path = self.paths[ob]
				for x, y, color, frame in path:
					click.append( [frame, "frame", Vector([x, y]), tuple(channels)] )

			self.click[ob] = click

		#context.preferences.edit.use_global_undo = global_undo

	except Exception as e:
		print(e)
		tb = sys.exc_info()[-1]
		print(traceback.extract_tb(tb))
		# restore global undo in case of failure (see T52524)
		#context.preferences.edit.use_global_undo = global_undo

# TODO: Skip world_to_screen for non-clickable coords?
point_vertex_shader = """
uniform mat4 ModelViewProjectionMatrix;

in vec2 pos;
in vec4 color;
in float radius;
in int flags; 

out vec2 _pos;
out vec4 _color;
out float _radius;
out int _flags;

void main()
{
	_radius = radius;
	_color = color;
	_pos = pos;
	_flags = flags;
	gl_Position = ModelViewProjectionMatrix * vec4(pos, 0.0, 1.0);
}
"""

# TODO: Should this code be further optimized?
point_frag_shader = """
uniform vec2 resolution;
uniform float outline_radius;
uniform float outline_blur;
//uniform vec3 outline_color;

in vec2 _pos;
in vec4 _color;
in float _radius;
flat in int _flags;

out vec4 FragColor;

float maprange(float oldmin, float oldmax, float newmin, float newmax, float val)
{
	float fac = (val - oldmin) / (oldmax - oldmin);
	return (newmin * (1.0 - fac) + newmax * fac);
}

float maprangeclamp(float oldmin, float oldmax, float newmin, float newmax, float val)
{
	val = clamp(val, oldmin, oldmax);
	return maprange(oldmin, oldmax, newmin, newmax, val);
}

void main()
{
	float maxr = max(resolution.x, resolution.y); // Resizing like this is consistent with how the view behaves when areas are resized
	float radius_corrected = _radius * (maxr / 500.0);
	float do_outline = float(_flags % 2);

	float dist = length(gl_FragCoord.xy - _pos);
	float o_radius = outline_radius * do_outline; 

	float color_fac = maprangeclamp(_radius, _radius + outline_blur, 0.0, 1.0, dist);
	float outline_fac = maprangeclamp(_radius + o_radius, _radius + o_radius + outline_blur, 0.0, 1.0, dist);
	vec4 alpha_corrected = mix(vec4(0.0, 0.0, 0.0, 0.0), _color, float(_color.a > 0.0));
	vec4 outline_color = vec4(0.0, 0.0, 0.0, 1.0);
    FragColor = mix(alpha_corrected, outline_color, color_fac);
	FragColor = vec4(FragColor.rgb, mix(FragColor.a, 0.0, outline_fac));
}
"""

colored_points_shader = gpu.types.GPUShader(point_vertex_shader, point_frag_shader)

tri_line_vertex_shader = """
uniform mat4 ModelViewProjectionMatrix;

in vec2 pos;
in vec4 color;
in vec3 wmo; // Width, max width, outline

out float width;
out flat vec2 maxw_outline;
out vec4 _color;

void main()
{
	_color = color;
	width = wmo.x;
	maxw_outline = wmo.yz;
	gl_Position = ModelViewProjectionMatrix * vec4(pos, 0.0, 1.0);
}
"""

tri_line_fragment_shader = """
#pragma BLENDER_REQUIRE(gpu_shader_colorspace_lib.glsl)

in float width;
in vec4 _color;
in vec2 maxw_outline;

uniform float blur;

out vec4 FragColor;

float maprange(float oldmin, float oldmax, float newmin, float newmax, float val)
{
	float fac = (val - oldmin) / (oldmax - oldmin);
	return (newmin * (1.0 - fac) + newmax * fac);
}

float maprangeclamp(float oldmin, float oldmax, float newmin, float newmax, float val)
{
	val = clamp(val, oldmin, oldmax);
	return maprange(oldmin, oldmax, newmin, newmax, val);
}

void main()
{
	float grad = maprangeclamp(maxw_outline.x-blur, maxw_outline.x, 0.0, 1.0, width);
	float maxtot = maxw_outline.x+maxw_outline.y;
	float gradoutline = maprangeclamp(maxtot-blur, maxtot, 1.0, 0.0, width);
	vec4 outline = mix(_color, vec4(0.0, 0.0, 0.0, 1.0), float(maxw_outline.y>0.0));
	FragColor = mix(_color, outline, grad);
	FragColor.a *= gradoutline;
	FragColor = blender_srgb_to_framebuffer_space(FragColor);
}
"""

tri_line_shader = gpu.types.GPUShader(tri_line_vertex_shader, tri_line_fragment_shader)

# TODO: Thickness multiplier that affects everything
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
	colors_base = [tuple(mt.point_color_loc), tuple(mt.point_color_rot), tuple(mt.point_color_scl)]
	zeroadd = [0.0, 0.0, 0.0, 0.0]
	zeromul = [1.0, 1.0, 1.0, 1.0]
	for c in chans:
		colors_add = [(colors_base[i] if c[i] else zeroadd) for i in range(3)]
		colors_mul = [(colors_base[i] if c[i] else zeromul) for i in range(3)]

		mulled = mul4(colors_mul[0], mul4(colors_mul[1], colors_mul[2]))
		added = add4(colors_add[0], add4(colors_add[1], colors_add[2]))

		final = lerp4(mt.handle_color_fac, added, mulled)

		colors_cooked[c] = final


	#uniform_line_shader = gpu.shader.from_builtin('3D_POLYLINE_UNIFORM_COLOR')
	colored_line_shader: gpu.types.GPUShader = gpu.shader.from_builtin('3D_POLYLINE_SMOOTH_COLOR')
	#colored_points_shader = gpu.shader.from_builtin('2D_FLAT_COLOR')
	colored_points_shader.uniform_float("ModelViewProjectionMatrix", bpy.context.region_data.perspective_matrix)
	colored_points_shader.uniform_float("resolution", Vector((bpy.context.area.width, bpy.context.area.height)))
	colored_points_shader.uniform_float("outline_radius", mt.point_outline_size)
	colored_points_shader.uniform_float("outline_blur", mt.point_outline_blur)

	radii = {"keyframe": mt.keyframe_size, "timebead": mt.timebead_size, "frame": mt.frame_size, "handle_left": mt.handle_size, "handle_right": mt.handle_size}
	maxr = max(radii.values())
	maxr += mt.point_outline_size + mt.highlight_size + mt.point_outline_blur
	gpu.state.point_size_set(maxr*2.0 + 3.0)
	gpu.state.blend_set("ALPHA")

	poss = []
	cols = []
	
	for_shader = ([], [], []) if mt.pretty_lines else [[], []]
	line_converter = line_to_tris if mt.pretty_lines else line_strip_to_lines
	
	if mt.path_style == 'simple':
		
		simple_color = mt.simple_color
		for ob, path in self.paths.items():
			for i, [x, y, color, frame] in enumerate(path):
				if frame < limit_min or frame > limit_max:
					continue
				poss.append(Vector((x, y)))
				cols.append(simple_color)
			line_converter(poss, cols, mt.path_width, mt.path_outline_width, for_shader)
			poss.clear()
			cols.clear()

	else:
		for ob, path in self.paths.items():
			for i, [x, y, color, frame] in enumerate(path):
				if frame < limit_min or frame > limit_max:
					continue
				poss.append(Vector((x, y)))
				cols.append(color)
			line_converter(poss, cols, mt.path_width, mt.path_outline_width, for_shader)
			poss.clear()
			cols.clear()

	if not mt.pretty_lines and mt.path_outline_width > 0.0:
		outline_for_shader = (for_shader[0].copy(), len(for_shader[1]))

	# Draw rotation spines
	if mt.show_spines:
		colored_line_shader.bind()
		colored_line_shader.uniform_float("lineWidth", 2)
		poss = []
		cols = []
		for ob_spines in self.spines.values():
			for frame, locs in ob_spines.items():
				if frame < limit_min or frame > limit_max:
					continue
				
				to_use = (mt.pXspines, mt.pYspines, mt.pZspines, mt.nXspines, mt.nYspines, mt.nZspines)
				to_use_colors = (mt.spine_x_color, mt.spine_y_color, mt.spine_z_color, mt.spine_x_color, mt.spine_y_color, mt.spine_z_color)
				for i in range(6):
					if to_use[i]:
						cols.append(to_use_colors[i])
						poss.append(Vector((locs[0][0], locs[0][1])))
						cols.append(to_use_colors[i])
						poss.append(Vector((locs[1][i][0], locs[1][i][1])))
						line_converter(poss, cols, mt.path_width, 0.0, for_shader) # TODO: spine width
						poss.clear()
						cols.clear()

	point_poss = []
	point_cols = []
	point_rads = [] # radii is correct but 3-letter thing ending in s is better
	point_flags = []

	if self.highlighted_coord:
		rad = radii[self.highlighted_coord[1]] + mt.point_outline_size + mt.highlight_size
		colored_points_shader.bind()

		point_poss.append(self.highlighted_coord[0])
		point_cols.append(mt.highlight_color)
		point_rads.append(rad)
		point_flags.append(mt.highlight_do_outline)

	# draw frames
	if mt.frame_display:
		for ob, path in self.paths.items():
			for x, y, color, frame in path:
				if frame < limit_min or frame > limit_max:
					continue

				point_poss.append((x, y))
				if self.active_frame and ob == self.active_frame[0] \
				and abs(frame - self.active_frame[1]) < 1e-4:
					point_cols.append(mt.selection_color)
				else:
					point_cols.append(mt.frame_color)
				point_rads.append(mt.frame_size)
				point_flags.append(True)

	# time beads are shown in timing mode
	if mt.mode == 'timing':
		for ob, values in self.timebeads.items():
			for frame, [coords, channels] in values.items():
				if frame < limit_min or frame > limit_max:
					continue

				point_poss.append((coords[0], coords[1]))
				if self.active_timebead and \
				ob == self.active_timebead[0] and \
				abs(frame - self.active_timebead[1]) < 1e-4:
					point_cols.append(mt.selection_color)
				else:
					point_cols.append(mt.timebead_color)
				point_rads.append(mt.timebead_size)
				point_flags.append(True)


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
						p1 = Vector((self.keyframes[ob][frame][0][0],
								self.keyframes[ob][frame][0][1]))
						p2 = Vector((coords[0], coords[1]))

						newp1, newp2 = chop_line(p1, p2, 
							mt.keyframe_size+mt.point_outline_size/2.0, 
							mt.handle_size+mt.point_outline_size/2.0) #? Performant?

						poss.append(newp1)
						poss.append(newp2)

						if self.active_handle and \
						ob == self.active_handle[0] and \
						side == self.active_handle[2] and \
						get_chani(self.active_handle[3]) == chan and \
						abs(frame - self.active_handle[1]) < 1e-4:
							cols.append(mt.selection_color_dark)
							cols.append(mt.selection_color_dark)
						else:
							cols.append(mt.handle_line_color)
							cols.append(mt.handle_line_color)

						line_converter(poss, cols, mt.path_width, 0.0, for_shader) # TODO: handle width?
						poss.clear()
						cols.clear()

		# draw handles
		for chan in range(3):
			for ob, values in self.handles.items():
				for frame, sides in values[chan].items():
					if frame < limit_min or frame > limit_max:
						continue
					
					for side, coords in sides.items():
						point_poss.append((coords[0], coords[1]))
						if self.active_handle and \
						ob == self.active_handle[0] and \
						side == self.active_handle[2] and \
						get_chani(self.active_handle[3]) == chan and \
						abs(frame - self.active_handle[1]) < 1e-4:
							point_cols.append(mt.selection_color)
						else:
							point_cols.append(colors_cooked[make_chan(chan)])
						point_rads.append(mt.handle_size)
						point_flags.append(True)

	# draw keyframes
	for ob, values in self.keyframes.items():
		for frame, [coords, channels] in values.items():
			if frame < limit_min or frame > limit_max:
				continue

			point_poss.append((coords[0], coords[1]))
			if self.active_keyframe and \
			ob == self.active_keyframe[0] and \
			abs(frame - self.active_keyframe[1]) < 1e-4:
				point_cols.append(mt.selection_color)
			else:
				point_cols.append(colors_cooked[channels])
			point_rads.append(mt.keyframe_size)
			point_flags.append(True)
	
	if mt.pretty_lines:
		tri_line_shader.bind()
		tri_line_shader.uniform_float("blur", 1.0)
		batch = batch_for_shader(tri_line_shader, 'TRIS', {"pos": for_shader[0], "color": for_shader[1], "wmo": for_shader[2]})
		batch.draw(tri_line_shader)
	else:
		if mt.path_outline_width > 0.0:
			colored_line_shader.uniform_float("lineWidth", mt.path_width + mt.path_outline_width * 2.0)
			batch = batch_for_shader(colored_line_shader, 'LINES', {"pos": outline_for_shader[0], "color": [(0.0, 0.0, 0.0, 1.0) for i in range(outline_for_shader[1])]})
			batch.draw(colored_line_shader)

		colored_line_shader.bind()
		colored_line_shader.uniform_float("lineWidth", mt.path_width)
		batch = batch_for_shader(colored_line_shader, 'LINES', {"pos": for_shader[0], "color": for_shader[1]})
		batch.draw(colored_line_shader)

	colored_points_shader.bind()
	batch = batch_for_shader(colored_points_shader, 'POINTS', {"pos": point_poss, "color": point_cols, "radius": point_rads, "flags": point_flags})
	batch.draw(colored_points_shader)


	# draw keyframe-numbers
	if mt.keyframe_numbers:
		blf.size(0, mt.keyframe_text_size, 72)
		blf.color(0, 1.0, 1.0, 0.0, 1.0)
		for ob, values in self.keyframes.items():
			for frame, [coords, channels]  in values.items():
				if frame < limit_min or frame > limit_max:
					continue
				blf.position(0, coords[0] + mt.keyframe_text_offset_x, coords[1] + mt.keyframe_text_offset_y, 0)
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

	# Draw drag UI
	if self.drag:
		constraint_colors = [\
			[[0.0, 0.0, 0.0, 1.0], [1.0, 0.1, 0.1, 1.0]],
			[[0.0, 0.0, 0.0, 1.0], [0.1, 1.0, 0.1, 1.0]],
			[[0.0, 0.0, 0.0, 1.0], [0.1, 0.1, 1.0, 1.0]]]

		constraint_texts = ["X", "Y", "Z"]
		orient_texts = ["(Orientation 1)", "(Orientation 2)"]
		chan_texts = ["L", "R", "S"]

		ob, frame, thing, chans = self.getactive()

		#TODO: less hardcoded text positions?
		blf.size(0, 12, 130)
		blf.position(0, 10, 40, 0)
		blf.color(0, *mt.text_color)
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

		if sum(chans) > 1:
			blf.size(0, 12, 130)
			blf.position(0, 10, 80, 0)
			blf.color(0, *mt.text_color)
			blf.draw(0, "Working on: ")
			chosen_chans = self.chosen_chans
			colors_noyes = [(0.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0)]
			for i in range(3):
				if not chans[i]:
					continue
				blf.position(0, 150 + i*30, 80, 0)
				blf.color(0, *colors_noyes[chosen_chans[i]])
				blf.draw(0, chan_texts[i])

	# restore opengl defaults
	gpu.state.point_size_set(1.0) # TODO: is this the correct value?
	# ? gpu.state.blend_set("ALPHA")
	# TODO: unbind shaders?

def swizzle_constraint(vec, constraint):
	"""Given a 2D vector and a constraint of kind (a, b, c) where 1 or 2 values are True, swizzle a new 3D vector from the respective coords"""

	newvec = Vector((0.0, 0.0, 0.0))
	coord = 0
	for i in range(3):
		if constraint[i]:
			newvec[i] = vec[coord]
			coord += 1
	
	return newvec

def is_constrained(constraint):
	return constraint[0] or constraint[1] or constraint[2] 

def get_keyframes(curves: list[FCurve], frame: float) -> list[tuple[int, Keyframe]]:
	"""Returns a list of (index, keyframe) for all the keyframes found in the list."""
	res = []
	i = 0
	for fcurve in curves:
		for kf in fcurve.keyframe_points:
			if abs(kf.co[0] - frame) < 0.0001:
				res.append((i, kf))
				break

		i += 1
	
	return res

# change data based on mouse movement
def drag(self, context: Context, event):
	mt: MotionTrailProps = context.window_manager.motion_trail
	
	ob, frame, extra, ori_chans = self.getactive()
	inverse_mat: Matrix = self.cache_inverse.get_matrix(frame, ob, context)
	#decomposed = inverse_mat.decompose()
	#inverse_mat = Matrix.LocRotScale(decomposed[0], decomposed[1], Vector((1.0, 1.0, 1.0)))
		
	mouse_ori_world = inverse_mat @ screen_to_world(context, self.drag_mouse_ori)
	transformed_diff = inverse_mat @ screen_to_world(context, self.drag_mouse_accumulate + self.drag_mouse_ori)

	d: Vector = transformed_diff - mouse_ori_world
	if is_constrained(self.constraint_axes):
		if self.constraint_orientation == 1: # Possibly add more?
			d = swizzle_constraint(self.drag_mouse_accumulate * 0.05, self.constraint_axes)
		else:
			d = d * Vector(self.constraint_axes)
	sensitivities = (mt.sensitivity_location, mt.sensitivity_rotation * 0.3, mt.sensitivity_scale * 0.7)
	d_2d = self.drag_mouse_accumulate

	all_curves = get_curves(ob)
	chosen_chans = self.chosen_chans

	def update_this_handle(kf: Keyframe, side: bool, dif: float, this_ori_kf):
		sides_type = ["handle_left_type", "handle_right_type"]
		sides = [kf.handle_left, kf.handle_right]
		other_side = 1 - side
		originals = [this_ori_kf[1], this_ori_kf[2]]

		if getattr(kf, sides_type[side]) in ('AUTO', 'AUTO_CLAMPED'):
			kf.handle_left_type = 'ALIGNED'
			kf.handle_right_type = 'ALIGNED'
		elif getattr(kf, sides_type[side]) == 'VECTOR':
			kf.handle_left_type = 'FREE'
			kf.handle_right_type = 'FREE'

		sides[side][1] = originals[side][1] + dif
		if getattr(kf, sides_type[side]) == 'ALIGNED':
			sides[other_side][1] = originals[other_side][1] - dif

	def quat_transform(oldd: list[float], quat_vals: list[float]):
		to_eul = Vector(Quaternion(quat_vals).to_euler())
		to_eul_added = to_eul + oldd
		new_quat = Euler(to_eul_added).to_quaternion()
		newd = Vector(new_quat) - Vector(quat_vals)
		return newd
	

	# Alter keyframe values and handle coordinates
	if mt.mode == 'values':

		chan = findlist(True, chosen_chans) # Only 1 channel is chooseable in values mode

		kfs = get_keyframes(all_curves[chan], frame)
		kf_ori = self.keyframes_ori[ob][chan]
		d_sens = d * sensitivities[chan]

		if self.op_type == 0 and self.active_keyframe: # If trying to grab a keyframe, move keyframe around

			if len(all_curves[chan]) == 4: # Deal with quaternions
				d_sens = quat_transform(d_sens, [kf_ori[fcurv][frame][0][1] for fcurv in range(4)]) # TODO: Potential exception when user is being a user and doesn't have 4 quaternion KFs?
			
			for fcurvi, kf in kfs:
				this_ori_kf = kf_ori[fcurvi][frame]
				if not mt.allow_negative_scale and (chan == 2) and (this_ori_kf[0][1] + d_sens[fcurvi] < 0):
					d_sens[fcurvi] = abs(this_ori_kf[0][1] + d_sens[fcurvi]) - this_ori_kf[0][1]

				kf.co[1] = this_ori_kf[0][1] + d_sens[fcurvi]
				kf.handle_left[1] = this_ori_kf[1][1] + d_sens[fcurvi]
				kf.handle_right[1] = this_ori_kf[2][1] + d_sens[fcurvi]

		elif (self.op_type == 0 and self.active_handle) or self.op_type == 1: #if trying to grab a handle, or if trying to rotate either, move keyframe handle/s
			if len(all_curves[chan]) == 4:
				d_sens = quat_transform(d_sens, [kf_ori[fcurvi][frame][0][1] for fcurvi in range(4)]) # ? Does this even work?

			for fcurvi, kf in kfs:
				this_ori_kf = kf_ori[fcurvi][frame]

				if not extra == "right":
					update_this_handle(kf, 0, d_sens[fcurvi], this_ori_kf)
				if not extra == "left":
					update_this_handle(kf, 1, d_sens[fcurvi], this_ori_kf)

				all_curves[chan][fcurvi].update()

		elif self.op_type == 2: #If trying to scale, scale keyframe handle/s    Is this if necessary?
			d_sens = d.copy() * 0.2
			
			do_left = not extra == "right"
			do_right = not extra == "left"

			if len(all_curves[chan]) == 4:
				d_sens = quat_transform(d_sens, [kf_ori[fcurvi][frame][0][1] for fcurvi in range(4)]) # ? How effective is this for scaling KFs?
				one = Vector((1, 1, 1, 1))
			else:
				one = Vector((1, 1, 1))

			d_sens = Vector(d_sens) + one
			if not mt.allow_negative_handle_scale:
				d_sens = vecabs(d_sens)
			
			for fcurvi, kf in kfs:
				this_ori_kf = kf_ori[fcurvi][frame]
				centre = Vector(this_ori_kf[0])
				dif_left = centre - Vector(this_ori_kf[1])
				dif_right = centre - Vector(this_ori_kf[2])

				if kf.handle_left_type in ('AUTO', 'AUTO_CLAMPED') or \
					kf.handle_right_type in ('AUTO', 'AUTO_CLAMPED'):
					kf.handle_left_type = 'ALIGNED'
					kf.handle_right_type = 'ALIGNED'

				if do_left: kf.handle_left[0], kf.handle_left[1] = centre - d_sens[fcurvi] * dif_left
				if do_right: kf.handle_right[0], kf.handle_right[1] = centre - d_sens[fcurvi] * dif_right

				all_curves[chan][fcurvi].update()

	# change position of all keyframes on timeline
	elif mt.mode == 'timing' and self.active_timebead: # TODO: potential code merge with below timing if?

		frame_ori = extra
		ranges = []
		for chan in range(len(ori_chans)):
			if not ori_chans[chan]:
				continue
			ranges.extend([val for c in all_curves[chan] for val in c.range()])

		ranges.sort() # TODO: Don't sort just for 2 values
		range_min = round(ranges[0])
		range_max = round(ranges[-1])
		range_len = range_max - range_min
		dx_screen = d_2d[0] / context.region.width * range_len
		new_frame = frame + dx_screen
		shift_low = max(1e-4, (new_frame - range_min) / (frame - range_min))
		shift_high = max(1e-4, (range_max - new_frame) / (range_max - frame))

		new_mapping = FloatMap()

		if mt.retime_old_y:
			cancel_drag(self, context)
			retimedy = [[], [], []]
			for chan in range(len(chosen_chans)):
				if not chosen_chans[chan]:
					continue

				for fcurvi, curve in enumerate(all_curves[chan]):
					retimedy[chan].append({})
					for j, kf in enumerate(curve.keyframe_points):
						frame_ori = kf.co[0]
						
						if frame_ori <= frame:
							frame_new = (frame_ori - range_min) * shift_low + range_min
						else:
							frame_new = range_max - (range_max - frame_ori) * shift_high
						
						frame_new = max(
									range_min + j, min(frame_new, range_max -
									(len(curve.keyframe_points) - j) + 1)
									)
						retimedy[chan][fcurvi][j] = curve.evaluate(frame_new)


		for chan in range(len(chosen_chans)):
			if not chosen_chans[chan]:
				continue

			for fcurvi, curve in enumerate(all_curves[chan]):
				for j, kf in enumerate(curve.keyframe_points):
					if not self.frame_map:
						frame_ori = kf.co[0]
					else:
						frame_ori = self.frame_map[kf.co[0]]
					
					if frame_ori <= frame:
						frame_new = (frame_ori - range_min) * shift_low + range_min
					else:
						frame_new = range_max - (range_max - frame_ori) * shift_high
					
					frame_new = max(
								range_min + j, min(frame_new, range_max -
								(len(curve.keyframe_points) - j) + 1)
								)
					d_frame = frame_new - frame_ori
					new_mapping[frame_new] = frame_ori
					kf.co[0] = frame_new
					kf_ori = self.keyframes_ori[ob][chan][fcurvi][frame_ori]
					kf.handle_left[0] = kf_ori[1][0] + d_frame
					kf.handle_right[0] = kf_ori[2][0] + d_frame
					if mt.retime_old_y:
						newy = retimedy[chan][fcurvi][j]
						ydiff = kf.co[1] - newy
						kf.co[1] = newy
						kf.handle_left[1] = kf_ori[1][1] - ydiff
						kf.handle_right[1] = kf_ori[2][1] - ydiff
				all_curves[chan][fcurvi].update()

		del self.frame_map
		self.frame_map = new_mapping

	# change position of active keyframe on the timeline
	elif mt.mode == 'timing' and self.active_keyframe:
		if not self.frame_map:
			frame_ori = frame
		else:
			frame_ori = self.frame_map[frame]

		bignum = 1e64 # TODO: is this big number good?
		margin = 0.1 # To prevent accidentally losing our chosen keyframe when the user drags too close

		range_both = [-bignum, bignum]
		kf_ob_ori = self.keyframes_ori[ob]
		for chan in range(len(chosen_chans)):
			if not chosen_chans[chan]:
				continue

			for fcurvi in range(len(kf_ob_ori[chan])):
				kfs_ori = self.keyframes_ori[ob][chan][fcurvi]
				for kf_frame, kf in kfs_ori.items():
					if kf_frame < (frame_ori - margin) and kf_frame > range_both[0]:
						range_both[0] = kf_frame
					if kf_frame > (frame_ori + margin) and kf_frame < range_both[1]:
						range_both[1] = kf_frame

		drag_min = -context.region.width
		drag_max = context.region.width
		drag_amt = d_2d[0]

		if range_both[0] != range_both[1]: # Drag when there's KFs around
		
			if range_both[0] == -bignum: # Free drag beyond min
				range_both[0] = frame_ori + (frame_ori - range_both[1]) + margin # + margin so it's centered
				range_both[1] -= margin
				if drag_amt > drag_max:
					drag_amt = drag_max
			elif range_both[1] == bignum: # Free drag beyond max
				range_both[0] += margin
				range_both[1] = frame_ori + (frame_ori - range_both[0]) + margin # + margin so it's centered
				if drag_amt < drag_min:
					drag_amt = drag_min
			else:
				range_both[0] += margin
				range_both[1] -= margin
				# Adjust drag range such that 0.0 drag amount gives the original frame
				fac = (frame_ori - range_both[0]) / (range_both[1] - range_both[0])
				drag_min = (drag_max * fac) / (fac - 1.0)
				drag_amt = clamp(drag_min, drag_max, d_2d[0])

		else: # Drag with no other KFs
			fps = context.scene.render.fps / context.scene.render.fps_base # When no KF a given side, drag will be based on scene fps
			range_both = [-2 * fps, 2 * fps]

		new_frame = maprange(drag_min, drag_max, range_both[0], range_both[1], drag_amt)
		d_frame = new_frame - frame_ori


		if mt.retime_old_y:
			cancel_drag(self, context)
			frame = extra
			retimedy = [[] for i in range(len(chosen_chans))]

			for chan in range(len(chosen_chans)):
				if not chosen_chans[chan]:
					continue
				
				kfs = get_keyframes(all_curves[chan], frame)
				for (fcurvi, kf) in kfs:
					retimedy[chan].append([])
					retimedy[chan][fcurvi] = all_curves[chan][fcurvi].evaluate(new_frame)

		for chan in range(len(chosen_chans)):
			if not chosen_chans[chan]:
				continue

			kfs = get_keyframes(all_curves[chan], frame)
			for (fcurvi, kf) in kfs:
				this_ori_kf = self.keyframes_ori[ob][chan][fcurvi][frame_ori]
				kf.co[0] = new_frame
				kf.handle_left[0] = this_ori_kf[1][0] + d_frame
				kf.handle_right[0] = this_ori_kf[2][0] + d_frame
				if mt.retime_old_y:
					newy = retimedy[chan][fcurvi]
					ydiff = this_ori_kf[0][1] - newy
					kf.co[1] = newy
					kf.handle_left[1] = this_ori_kf[1][1] + ydiff
					kf.handle_right[1] = this_ori_kf[2][1] + ydiff
				all_curves[chan][fcurvi].update()
		
		del self.frame_map # TODO: is this necessary?
		self.frame_map = FloatMap()
		self.frame_map[new_frame] = frame_ori

		self.active_keyframe = (ob, new_frame, frame_ori, ori_chans)

	return


# revert changes made by dragging
def cancel_drag(self, context):
	mt: MotionTrailProps = context.window_manager.motion_trail
	ob, frame, extra, ori_chans = self.getactive()
	chosen_chans = self.chosen_chans
	all_curves = get_curves(ob)

	# TODO: Merge these 2 ifs to simplify code
	# revert change in values of active keyframe and its handles
	if mt.mode == 'values':
		chan = findlist(True, chosen_chans) # Only 1 channel is chooseable in values mode
		kfs = get_keyframes(all_curves[chan], frame)
		for fcurvi, kf in kfs:
			this_ori_kf = self.keyframes_ori[ob][chan][fcurvi][frame]
			kf.co[1] = this_ori_kf[0][1]
			kf.handle_left[0], kf.handle_left[1] = this_ori_kf[1] # TODO: Is writing it like this necessary? I feel assigning a list directly may not work. Test.
			kf.handle_right[0], kf.handle_right[1] = this_ori_kf[2]
			kf.handle_left_type = this_ori_kf[3]
			kf.handle_right_type = this_ori_kf[4]
			all_curves[chan][fcurvi].update()

	# revert position of all keyframes and handles on timeline
	elif mt.mode == 'timing':
		
		for chan in range(len(chosen_chans)):
			if not chosen_chans[chan]:
				continue

			for fcurvi, curve in enumerate(all_curves[chan]):
				for kf in curve.keyframe_points:
					if not self.frame_map.exists(kf.co[0]):
						continue

					frame = self.frame_map[kf.co[0]]
					
					this_ori_kf = self.keyframes_ori[ob][chan][fcurvi][frame]
					kf.co[0], kf.co[1] = this_ori_kf[0] # See above TODO
					kf.handle_left[0], kf.handle_left[1] = this_ori_kf[1]
					kf.handle_right[0], kf.handle_right[1] = this_ori_kf[2]
					kf.handle_left_type = this_ori_kf[3]
					kf.handle_right_type = this_ori_kf[4]
				all_curves[chan][fcurvi].update()
				
		if self.active_keyframe:
			self.active_keyframe = (ob, extra, extra, ori_chans)

	del self.frame_map
	self.frame_map = FloatMap()

	return


# return the handle type of the active selection
def get_handle_type(self, active_keyframe, active_handle):
	if active_keyframe:
		ob, frame, side, chans = active_keyframe
		side = "both"
	elif active_handle:
		ob, frame, side, chans = active_handle
	else:
		# no active handle(s)
		return (False)

	all_curves = get_curves(ob)
	for chan in range(len(all_curves)):
		if not chans[chan]:
			continue
		for fcurv in range(len(all_curves[chan])):
			for kf in all_curves[chan][fcurv].keyframe_points:
				if kf.co[0] == frame:
					if side in ("left", "both"):
						return (kf.handle_left_type)
					else:
						return (kf.handle_right_type)

	return ("AUTO")


# Turn the given frame into a keyframe
def insert_keyframe(frame: float, ob: Object, chans: list[bool]):
	all_curves = get_curves(ob)
	new_fcurve_paths = ["location", "rotation_euler", "scale"]
	for chan in range(len(chans)):
		if not chans[chan]:
			continue

		for fcurvi in range(len(all_curves[chan])):
			c = all_curves[chan][fcurvi]
			y = c.evaluate(frame)
			if not c.keyframe_points:
				if len(all_curves[chan]) == 4:
					data_path = "rotation_quaternion"
				else:
					data_path = new_fcurve_paths[chan]
				c = ob.animation_data.action.fcurves.new(data_path, index=fcurvi)
			
			c.keyframe_points.insert(frame, y)

def handle_update(self, context):
	mt: MotionTrailProps = context.window_manager.motion_trail
	mt.handle_update = True

# change the handle type of the active selection
def set_handle_type(self, context):
	self: MotionTrailOperator
	mt: MotionTrailProps = context.window_manager.motion_trail

	if not mt.handle_type_enabled:
		return

	if self.handle_type_old == mt.handle_type:
		# function called because of selection change, not change in type
		return

	if self.active_keyframe:
		ob, frame, side, chans = self.active_keyframe
		side = "both"
	elif self.active_handle:
		ob, frame, side, chans = self.active_handle
	else:
		# no active handle(s)
		return (False)
	new_type = mt.handle_type

	side = "both"

	all_curves = get_curves(ob)
	for chan in range(len(chans)):
		if not chans[chan]:
			continue

		for fcurvi in range(len(all_curves[chan])):
			for kf in all_curves[chan][fcurvi].keyframe_points:
				if kf.co[0] == frame:
					if side in ("left", "both"):
						kf.handle_left_type = new_type
					if side in ("right", "both"):
						kf.handle_right_type = new_type
					break
			all_curves[chan][fcurvi].update()
			

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
	_msgbus_owner_kf = None
	_timer = None

	drag: bool 
	"""Whether or not we're dragging"""
	lock: bool
	"""Whether or not we're changing the motion trail"""

	op_type: int = -1
	"""0 = grab (location), 1 = rotate, 2 = scale"""
	constraint_axes: list[bool] = [False, False, False]
	"""Bools for which axes are constrained. Please keep 3 long."""
	constraint_orientation: bool = 0
	"""0/False = Global, 1/True = Local"""

	click: dict[Object, list[any]]
	"""Items that may be clicked on. Structure: {ob: [[frame, type, coord, channels], ...], ob2: ...}"""

	keyframes_ori: dict[Object, list[list[dict[float, list[list[float]]]]]]
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

	highlighted_coord: list[float]
	"""Coordinates of highlighted point, for highlight on hover."""

	drag_mouse_ori: Vector
	"""Mouse position at start of drag"""

	drag_mouse_accumulate: Vector
	"""Accumulated mouse position from dragging, nicely factoring in shift/alt"""

	chosen_chans: tuple[bool, bool, bool]

	def getactive(self):
		if self.active_keyframe: return self.active_keyframe
		if self.active_handle: return self.active_handle
		if self.active_timebead: return self.active_timebead
		if self.active_frame: return self.active_frame
		return None

	@staticmethod
	def handle_add(self, context):

		if not context.window_manager.motion_trail.use_depsgraph:
			global global_mtrail_handler_calc
			global_mtrail_handler_calc = \
			MotionTrailOperator._handle_calc = bpy.types.SpaceView3D.draw_handler_add(
				calc_callback, (self, context), 'WINDOW', 'POST_VIEW')

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
			MotionTrailOperator._msgbus_owner_kf = global_mtrail_msgbus_owner
		
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

		if MotionTrailOperator._msgbus_owner_kf is not None:
			try:
				bpy.msgbus.clear_by_owner(MotionTrailOperator._msgbus_owner_kf)
			except:
				pass
			
		MotionTrailOperator._handle_calc = None
		MotionTrailOperator._handle_draw = None
		MotionTrailOperator._handle_update = None

	def caches_set(self, mt):
		getter = get_matrix_any_depsgraph if mt.use_depsgraph else get_matrix_any_custom_eval
		self.cache = MatrixCache(getter)
		getter_inverse = get_inverse_parents_depsgraph if mt.use_depsgraph else get_inverse_parents
		self.cache_inverse = MatrixCache(getter_inverse)


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

		if mt.handle_update:
			set_handle_type(self, context)
			mt.handle_update = False

		if mt.cache_change:
			mt.cache_change = False
			if mt.use_depsgraph:
				bpy.ops.ed.undo_push(message="Motion trail enabled depsgraph")
			self.caches_set(mt)
			self.handle_remove()
			self.handle_add(self, context)
			mt.force_update = True
			calc_callback(self, context)

		if mt.force_deselect:
			self.active_frame = False
			self.active_handle = False
			self.active_keyframe = False
			self.active_timebead = False
			mt.force_deselect = False

		if mt.use_depsgraph:
			calc_callback(self, context)

		#if not context.area or not context.region: or event.type == 'NONE':
			#context.area.tag_redraw()
		#	return {'PASS_THROUGH'}

		no_passthrough = False
		if self.drag:
			no_passthrough = True

			# TODO: Should this cancel_drag as well? could merge both ifs if so
			if (not context.active_object or
					context.active_object.mode not in ('OBJECT', 'POSE')):
				self.drag = False
				mt.force_update = True

			if (event.type in ['ESC', 'RIGHTMOUSE']):
				# cancel drag
				context.window.cursor_set('DEFAULT')
				self.drag = False
				mt.force_update = True
				cancel_drag(self, context)

			if event.type in ['X', 'Y', 'Z'] and event.value == 'PRESS':
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

			if event.type in self.transform_keys and event.value == 'PRESS':
				cancel_drag(self, context)

				id = findlist(event.type, self.transform_keys)
				ob, frame, extra, chans = self.getactive()

				if mt.mode == 'timing':
					if sum(chans) > 1: # No flip if single-chan
						self.chosen_chans = flip_chan(chans, self.chosen_chans, id)
				else:
					if not event.shift:
						self.op_type = id
					else:
						self.chosen_chans = single_chan(chans, id)
						
				drag(self, context, event)

			if event.type == 'MOUSEMOVE':
				# drag

				currmouse = Vector((event.mouse_x, event.mouse_y))
				prevmouse = Vector((event.mouse_prev_x, event.mouse_prev_y))

				# Move cursor back inside 3d viewport
				area = context.area
				maxx = area.x + area.width
				maxy = area.y + area.height
				margin = 5
				if event.mouse_x < (area.x + margin):
					context.window.cursor_warp(maxx-1-margin, event.mouse_y)
				if event.mouse_x > (maxx - margin):
					context.window.cursor_warp(area.x+1+margin, event.mouse_y)
				if event.mouse_y < (area.y + margin):
					context.window.cursor_warp(event.mouse_x, maxy-1-margin)
				if event.mouse_y > (maxy - margin):
					context.window.cursor_warp(event.mouse_x, area.y+1+margin)

				sens = 1.0
				if event.alt:
					sens = mt.sensitivity_alt
				if event.shift:
					sens = mt.sensitivity_shift

				self.drag_mouse_accumulate += (currmouse - prevmouse) * sens

				drag(self, context, event)

			if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
				# finish drag
				get_handle_type(self, self.active_keyframe, self.active_handle) # TODO: not a complete fix for panel UI not updating fast
				if self.active_keyframe:
					ob, new_frame, frame_ori, ori_chans = self.active_keyframe
					self.active_keyframe = (ob, new_frame, new_frame, ori_chans)
				context.window.cursor_set('DEFAULT')
				self.drag = False
				mt.force_update = True
				bpy.ops.ed.undo_push(message="Confirmed Motion Trail drag")

		else:

			if not mouse_in_region(event, context): return {'PASS_THROUGH'}

			if (event.type in self.transform_keys and event.value == 'PRESS' and self.getactive()):
				self.op_type = findlist(event.type, self.transform_keys)
				self.chosen_channel = 0
				context.window.cursor_set('SCROLL_XY')

				if self.active_frame:
					ob, frame, other, chans = self.active_frame
					insert_keyframe(frame, ob, chans) # TODO: transforms selector for inserting keyframes
					mt.force_update = True
					calc_callback(self, context) 
						
					self.active_keyframe = self.active_frame
					self.active_frame = False
					bpy.ops.ed.undo_push(message="Motion Trail added keyframes")
				
				ob, frame, other, chans = self.getactive()

				if mt.mode == 'timing':
					self.chosen_chans = chans
				else:
					self.chosen_chans = single_chan(chans, 0)

				self.loc_ori_ws = self.cache.get_location(frame, ob, context)
				self.frame_map = FloatMap()
				self.keyframes_ori = get_original_animation_data(context)
				self.drag_mouse_ori = Vector([event.mouse_region_x, event.mouse_region_y])
				self.drag_mouse_accumulate = Vector((0, 0))
				self.drag = True
				self.highlighted_coord = False

				self.constraint_axes = [False, False, False]
				self.constraint_orientation = False
				no_passthrough = True
		
			elif (event.type in [mt.select_key, mt.deselect_nohit_key] and event.value == 'PRESS') or \
				event.type == 'MOUSEMOVE':
				# Select or highlight
				clicked = Vector([event.mouse_region_x, event.mouse_region_y])

				#context.window_manager.motion_trail.force_update = True
				#TODO: Stare at the above line, should it be commented out?

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

				found = False
				for ob, values in self.click.items():
					if found:
						break
					for frame, type, coord, channels in values:
						if frame < frame_min or frame > frame_max:
							continue
						if (coord - clicked).length <= mt.select_threshold:
							found = True
							self.highlighted_coord = (coord, type)

							if event.type == mt.select_key and event.value == 'PRESS':
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
					self.highlighted_coord = False
					if event.type == mt.deselect_nohit_key and event.value == 'PRESS':
						# If a change happens (aka there was active, now there isn't), then no passthrough
						no_passthrough = self.getactive() and not mt.deselect_passthrough

						for attr in ["active_keyframe", "active_handle", "active_timebead", "active_frame"]:
							setattr(self, attr, False)
						mt.handle_type_enabled = False
						
				else:
					handle_type = get_handle_type(self, self.active_keyframe,
						self.active_handle)
					if handle_type:
						mt.handle_type_old = handle_type
						mt.handle_type = handle_type
					else:
						mt.handle_type_enabled = False
		

			elif event.type == mt.deselect_always_key and event.value == 'PRESS':
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
		kms = [
			bpy.context.window_manager.keyconfigs.active.keymaps['3D View'],
			bpy.context.window_manager.keyconfigs.active.keymaps['Object Mode']
			]
		kmis = []
		for km in kms:
			for kmi in km.keymap_items:
				# ??? "and not kmi.properties.texture_space" - why?
				if kmi.map_type == 'KEYBOARD':
					if kmi.idname == "transform.translate" :
						# ? kmis.append(kmi) - ???
						translate_key = kmi.type

					if kmi.idname == "transform.rotate":
						rotate_key = kmi.type
				
					if kmi.idname == "transform.resize": # ! Why the hell did they call it resize and not scale?!
						scale_key = kmi.type

		self.transform_keys = [translate_key, rotate_key, scale_key]

		mt: MotionTrailProps = context.window_manager.motion_trail

		if not mt.enabled:
			# enable
			self.active_keyframe = False
			self.active_handle = False
			self.active_timebead = False
			self.active_frame = False
			self.click = {}
			self.drag = False
			self.perspective = context.region_data.perspective_matrix
			self.displayed = []
			self.paths = {}
			self.keyframes = {}
			self.handles = {}
			self.timebeads = {}
			self.spines = {} 
			self.constraint_axes = [False, False, False]
			self.constraint_orientation = False
			self.affect_all_channels = False
			self.handle_type_old = False

			self.highlighted_coord = None
			self.last_frame = -1
			self.frame_map = FloatMap()
			self.chosen_chans = (False, False, False)

			mt.force_update = True
			mt.handle_type_enabled = False
			self.caches_set(mt)

			MotionTrailOperator.handle_add(self, context)
			mt.enabled = True

			if context.area:
				context.area.tag_redraw()

			context.window_manager.modal_handler_add(self)
			bpy.ops.ed.undo_push(message="Started Motion Trail modal operator")

			self._timer = wm.event_timer_add(0.0, window = context.window) # ! Can't undo while this is active
			#TODO: Workaround?                                                 ^
			return {'RUNNING_MODAL'}

		else:
			# disable
			MotionTrailOperator.handle_remove()
			mt.enabled = False

			if context.area:
				context.area.tag_redraw()

			if self._timer:
				context.window_manager.event_timer_remove(self._timer)

			return {'FINISHED'}

	def cancel(self, context):
		if self._timer:
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
	
	def draw(self, context):
		layout = self.layout
		col = layout.column()
		col.label(text="Are you sure?")
		col.label(text="If not, click off of this dialog box, or ESCape.")
	
	def execute(self, context):
		load_defaults(context)
		return {'FINISHED'}

	def invoke(self, context, event):
		wm = context.window_manager
		return wm.invoke_props_dialog(self)

def save_defaults(context):
	prefs = context.preferences.addons[__name__].preferences
	flat_props = flatten(configurable_props)
	for p in flat_props:
		current = getattr(context.window_manager.motion_trail, p)
		setattr(prefs.default_trail_settings, p, current)

	#bpy.ops.wm.save_userpref()

# TODO: Saved data doesn't persist after closing blender?!
class MotionTrailSaveDefaults(bpy.types.Operator):
	bl_idname="view3d.motion_trail_save_defaults"
	bl_label="Save Defaults"
	bl_description="Overwrite the defaults in the addon's preferences with what the current settings are.\nThis will also save the user preferences"

	def draw(self, context):
		layout = self.layout
		col = layout.column()
		col.label(text="Are you sure?")
		col.label(text="This will also save the user preferences!")
		col.label(text="If not, click off of this dialog box, or ESCape.")
	
	def execute(self, context):
		save_defaults(context)
		bpy.ops.wm.save_userpref()
		return {'FINISHED'}

	def invoke(self, context, event):
		wm = context.window_manager
		return wm.invoke_props_dialog(self)

class MotionTrailSaveUserpref(bpy.types.Operator):
	bl_idname="view3d.motion_trail_save_userpref"
	bl_label="Save Preferences"
	bl_description="Save the current user preferences"

	def draw(self, context):
		layout = self.layout
		col = layout.column()
		col.label(text="Are you sure?")
		col.label(text="This will also save any other changes done to the user preferences.")
		col.label(text="If not, click off of this dialog box, or ESCape.")
	
	def execute(self, context):
		bpy.ops.wm.save_userpref()
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
			col = box.column()
			col.prop(mt, "timebeads")
			col.prop(mt, "retime_old_y")
		if mt.mode == 'values':
			col = box.column()
			col.prop(mt, "allow_negative_scale")
			#col.prop(mt, "allow_negative_handle_scale")

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
			width_row = grouped.row(align=True)
			width_row.prop(mt, "path_width", text="Width")
			width_row.prop(mt, "path_outline_width", text="Outline")
			step_row = grouped.row(align=True)
			step_row.prop(mt, "path_step")
			step_row.prop(mt, "path_step_drag")
			row = grouped.row(align=True)
			row.prop(mt, "path_before")
			row.prop(mt, "path_after")
			col = col.column(align=True)
			col.prop(mt, "keyframe_numbers")
			if mt.keyframe_numbers:
				text_col_row = col.row(align=True)
				text_col_row.prop(mt, "text_color", text="Color")
				text_col_row.prop(mt, "selected_text_color", text="Selected")
				col.row(align=True).prop(mt, "keyframe_text_size")
				offset_row = col.row(align=True)
				offset_row.prop(mt, "keyframe_text_offset_x")
				offset_row.prop(mt, "keyframe_text_offset_y")

			col.prop(mt, "frame_display")
			if mt.frame_display:
				col.row().prop(mt, "frame_color")
				col.row().prop(mt, "frame_size")

			# Spines
			col.row().prop(mt, "show_spines")
			if mt.show_spines:
				pSpineStrings = ["pXspines", "pYspines", "pZspines"]
				nSpineStrings = ["nXspines", "nYspines", "nZspines"]
				spineColorStrings = ["spine_x_color", "spine_y_color", "spine_z_color"]
				
				spine_do_row = col.row()
				spine_do_row.prop(mt, "spine_do_rotation", text="Use rotation")
				spine_do_row.prop(mt, "spine_do_scale", text="Use scale")

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

		box = self.layout.box()
		col = box.column(align=True)
		if mt.mode == 'values':
			col.prop(mt, "handle_display",
				text="Handles")
			if mt.handle_display:
				row = col.row()
				row.enabled = mt.handle_type_enabled
				row.prop(mt, "handle_type")
				col.prop(mt, "handle_direction", text="Direction")
				sizes_row = col.row(align=True)
				sizes_row.prop(mt, "handle_length", text="Length")
				sizes_row.prop(mt, "handle_size", text="Size")
				
				handle_cols_row = col.row(align=True)
				handle_cols_row.prop(mt, "handle_line_color", text="Line")
				handle_cols_row.prop(mt, "selection_color_dark", text="Selected")
		else:
			col.row().prop(mt, "timebead_color")
			col.row().prop(mt, "timebead_size")


		box = self.layout.box()
		col = box.column()
		row = col.row()
		if mt.generic_colors_display: # TODO: make a function for this?
			row.prop(mt, "generic_colors_display", icon="DOWNARROW_HLT", text="", emboss=False)
		else:
			row.prop(mt, "generic_colors_display", icon="RIGHTARROW", text="", emboss=False)

		row.label(text="Generic color options")

		if mt.generic_colors_display:
			col.prop(mt, "pretty_lines")
			col.prop(mt, "keyframe_size")
			col.prop(mt, "point_outline_size")
			col.prop(mt, "point_outline_blur")
			handle_color_row = col.row()
			handle_color_row.prop(mt, "point_color_loc", text="Loc")
			handle_color_row.prop(mt, "point_color_rot", text="Rot")
			handle_color_row.prop(mt, "point_color_scl", text="Scale")

			handle_fac_row = col.row()
			handle_fac_row.prop(mt, "handle_color_fac")
			col.row().prop(mt, "selection_color")
			col.row().prop(mt, "highlight_color")
			col.row().prop(mt, "highlight_size")
			col.row().prop(mt, "highlight_do_outline")

		box = self.layout.box()
		col = box.column()
		row = col.row()

		if mt.controls_display:
			row.prop(mt, "controls_display", icon="DOWNARROW_HLT", text="", emboss=False)
		else:
			row.prop(mt, "controls_display", icon="RIGHTARROW", text="", emboss=False)

		row.label(text="Control options")

		if mt.controls_display:
			col.row().label(text="Sensitivty:")
			sens_row_lrs = col.row(align=True)
			sens_row_lrs.prop(mt, "sensitivity_location", text="Loc")
			sens_row_lrs.prop(mt, "sensitivity_rotation", text="Rot")
			sens_row_lrs.prop(mt, "sensitivity_scale", text="Scale")
			sens_row_modkeys = col.row(align=True)
			sens_row_modkeys.prop(mt, "sensitivity_shift")
			sens_row_modkeys.prop(mt, "sensitivity_alt")
			col.label(text="")
			col.row().prop(mt, "select_key", text="Select key")
			col.row().prop(mt, "select_threshold")
			col.row().prop(mt, "deselect_nohit_key")
			col.row().prop(mt, "deselect_always_key")
			col.row().prop(mt, "deselect_passthrough")
			col.label(text="For the time being, confirm/cancel")
			col.label(text="is LMB/RMB or Esc")
		
		box = self.layout.box()
		col = box.column()
		row = col.row()

		if mt.saveload_display:
			row.prop(mt, "saveload_display", icon="DOWNARROW_HLT", text="", emboss=False)
		else:
			row.prop(mt, "saveload_display", icon="RIGHTARROW", text="", emboss=False)

		row.label(text="Save/load")

		if mt.saveload_display:
			col.operator("view3d.motion_trail_load_defaults")
			col.operator("view3d.motion_trail_save_defaults")
			#col.operator("view3d.motion_trail_save_userpref")

DESELECT_WARNING = "Deselection will happen before your click registers to the rest of Blender.\n" +\
	"This can prevent you from changing the handle type if it's set to left click"

class MotionTrailProps(bpy.types.PropertyGroup):
	def internal_update(self, context):
		context.window_manager.motion_trail.force_update = True
		if context.area: # TODO: is this necessary?
			context.area.tag_redraw()

	def restart_operator(self, context: Context):
		if self.enabled:
			context.window_manager.motion_trail.cache_change = True

	def internal_update_deselect(self, context):
		context.window_manager.motion_trail.force_update = True
		context.window_manager.motion_trail.force_deselect = True

	# internal use
	enabled: BoolProperty(default=False)

	cache_change: BoolProperty(default=False)

	loaded_defaults: BoolProperty(default=False)

	force_update: BoolProperty(name="internal use",
		description="Force calc_callback to fully execute",
		default=False)
	"""Force calc_callback to fully execute, clearing its cache. Expensive AND will freeze animations when DG is involved!"""

	force_deselect: BoolProperty(
		default=False
	)
	"""Force the motion trail to deselect everything"""
		
	handle_update: BoolProperty(default=False)
	"""Tell the operator itself to update the handles."""

	handle_type_enabled: BoolProperty(default=False)

	# visible in user interface
	use_depsgraph: BoolProperty(name="Use depsgraph",
		description="Whether to use the depsgraph or not.\nUsing the depsgraph currently has the following ups and downs:\n\n+ Completely accurate motion trails that factor in all constraints, drivers, and so on.\n\n- Constantly resets un-keyframed changes to objects with keyframes.\n- Causes playback to pause when calculating (e.g. while dragging), due to which also...\n- Does not update with the graph editor or others.\n- Less performant",
		default=False,
		update=restart_operator
		)
	retime_old_y: BoolProperty(name="Position on old Y",
		description="Whether to put keyframes on the Y of the old f-curve rather than keeping their current Y.\nMay worsen drag performance",
		default=False
		)

	frame_display: BoolProperty(name="Frames",
			description="Display individual frames as manipulateable dots.\nClick and drag on one to make a new keyframe",
			default=False,
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
			update=handle_update
			)
	keyframe_numbers: BoolProperty(name="Keyframe numbers",
			description="Display keyframe numbers",
			default=False,
			update=internal_update
			)
	mode: EnumProperty(name="Mode", items=(
			("values", "Values", "Alter values of the keyframes and coordinates of their handles"),
			("timing", "Timing", "Change position of keyframes on timeline")),
			description="Enable editing of certain properties in the 3d-view",
			default='values',
			update=internal_update_deselect
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
			description="Step size for the frames the motion trail consists of.\nIncrease to improve uncached playback performance",
			default=1,
			min=1,
			max=10,
			update=internal_update
			)
	path_step_drag: IntProperty(name="Drag Step",
			description="Step size for the frames the motion trail consists of while dragging.\nIncrease to improve dragging performance",
			default = 1,
			min = 1,
			soft_max = 30,
			update=internal_update
			)
	path_style: EnumProperty(name="Path style", items=(
			("acceleration", "Acceleration", "Gradient based on relative acceleration."),
			("simple", "Simple", "A line with a single color."),
			("speed", "Speed", "Gradient based on relative speed.")),
			description="Information conveyed by path color",
			default='simple',
			update=internal_update
			)
	path_width: FloatProperty(name="Path width",
			description="Width in pixels",
			default=2.0,
			min=0.0,
			soft_max=8.0
			)
	path_outline_width: FloatProperty(name="Path outline width",
			description="Width of the outline around the path, 0 for none",
			default=0.0,
			min=0.0,
			soft_max=8.0
			)
	pretty_lines: BoolProperty(name="Pretty lines",
			description="Draw lines that properly connect between each segment. Noticeable with high widths and strong bends in the motion trail.\nMore performance intensive",
			default=False
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
	handle_size: FloatProperty(name="Handle size",
			description="Radius of the point of a handle in pixels",
			default=2.0,
			step=0.5
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
	keyframe_size: FloatProperty(name="Keyframe size",
			description="Radius in pixels for keyframe points",
			default=4.0,
			min=0.0,
			step=0.5
			)
	frame_size: FloatProperty(name="Frame size",
			description="Radius in pixels for individual frame points",
			default=1.75,
			min=0.0,
			step=0.5
			)
	timebead_size: FloatProperty(name="Bead size",
			description="Radius in pixels for timebeads",
			default=3.0,
			min=0.0,
			step=0.5
			)
	point_outline_size: FloatProperty(name="Point outline size",
			description="Radius in pixels for point outlines, extending past the filled color",
			default=1.75,
			min=0.0,
			step=0.5
			)
	point_outline_blur: FloatProperty(name="Point blur",
			description="Amount of pixels that the outlines and point colors will gradually fade for",
			default=1.0,
			min=0.01, # 0.0 breaks range mapping in the shader (min=max) and causes it to flip its output, making the points invisible
			step=0.1
			)
	
	saveload_display: BoolProperty(name="Save/load",
			description="Display save and load options",
			default=True
	)

	do_location: BoolProperty(name="Do Location",
			description="Show and work with location keyframes",
			default=True,
			update=internal_update_deselect
			)
	do_rotation: BoolProperty(name="Do Rotation",
			description="Show and work with rotation keyframes",
			default=False,
			update=internal_update_deselect
			)
	do_scale: BoolProperty(name="Do Scale",
			description="Show and work with scale keyframes",
			default=False,
			update=internal_update_deselect
			)

	allow_negative_scale: BoolProperty(name="Negative scaling",
			description="Whether to allow scale keyframes to get negative values or not",
			default=False
			)
	allow_negative_handle_scale: BoolProperty(name="Negative handle scaling",
			description="Whether to allow scaling handles negatively or not",
			default=False,
			)

	#Key stuff
	controls_display: BoolProperty(name="Control options",
			description="Display options related to controls",
			default=True
			)
	
	sensitivity_location: FloatProperty(name="Location sensitivity",
			description="Sensitivity for location-related values",
			default = 1.0
			)
	sensitivity_rotation: FloatProperty(name="Rotation sensitivity",
			description="Sensitivity for rotation-related values",
			default = 1.0
			)
	sensitivity_scale: FloatProperty(name="Scale sensitivity",
			description="Sensitivity for scale-related values",
			default = 1.0
			)
	sensitivity_shift: FloatProperty(name="Shift sensitivity",
			description="Sensitivity while holding shift",
			default = 0.35
			)
	sensitivity_alt: FloatProperty(name="Alt sensitivity",
			description="Sensitivity while holding alt",
			default = 3.0
			)
			
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
			description="Pressing this key will always deselect.\n" + DESELECT_WARNING,
			items=(
			("LEFTMOUSE", "Left Mouse Button", ""),
			("RIGHTMOUSE", "Right Mouse Button", ""),
			("NONE", "None", ""),),
			default="NONE"
			)
	select_threshold: FloatProperty(name="Selection distance",
			description="Distance in pixels for selecting something.\nSetting this too high will make the select button you've chosen unusable in the 3D viewport.\nIf this happens, deselect all objects, and set this back to a sane value",
			default=10.0,
			step=2,
			min=0.0,
			soft_max=50.0
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
			description="How many frames to step across for each spine, higher = less spines.\nThis value will be ignored if the path's step size is higher.\nWhen effective, if this is a value that's not a factor of the path's step size, performance will worsen",
			default=1,
			min=1,
			soft_max=10,
			update=internal_update,
			)

	spine_do_rotation: BoolProperty(name="Spines Use Rotation",
			description="Whether the spines will rotate themselves with the object's rotation",
			default=True,
			update=internal_update
			)
	spine_do_scale: BoolProperty(name="Spines Use Scale",
			description="Whether the spines will scale their length with the object's scale",
			default=False,
			update=internal_update
			)

	generic_colors_display: BoolProperty(name="Generic Colors Display",
			description="Whether to display the generic colors dropdown menu or not",
			default=True
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

	point_color_loc: FloatVectorProperty(name="Location point color",
			description="Color that unselected location handles and keyframes will be colored in",
			default=(1.0, 0.1, 0.1, 1.0),
			min=0.0, soft_max=1.0,
			size=4,
			subtype='COLOR'
			)
	point_color_rot: FloatVectorProperty(name="Rotation point color",
			description="Color that unselected rotation handles and keyframes will be colored in",
			default=(0.1, 1.0, 0.1, 1.0),
			min=0.0, soft_max=1.0,
			size=4,
			subtype='COLOR'
			)
	point_color_scl: FloatVectorProperty(name="Scale point color",
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
			description="Color that timebeads (dots in timing mode) will be colored in",
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
	keyframe_text_size: FloatProperty(name="Keyframe text size",
			description="What size keyframe numbers are",
			default=12.0,
			min=0.0,
			step=1.0
			)
	keyframe_text_offset_x: FloatProperty(name="Keyframe text offset X",
			description="How shifted text is from the keyframe itself, on the X axis",
			default=9.0,
			step=1.0
			)
	keyframe_text_offset_y: FloatProperty(name="Keyframe text offset Y",
			description="How shifted text is from the keyframe itself, on the Y axis",
			default=5.0,
			step=1.0
			)

	spine_x_color: FloatVectorProperty(name="X color",
			description="Color that spines corresponding to X rotation will be colored in",
			default=(0.7, 0.0, 0.0, 1.0),
			min=0.0, soft_max=1.0,
			size=4,
			subtype='COLOR'
			)
	spine_y_color: FloatVectorProperty(name="Y color",
			description="Color that spines corresponding to Y rotation will be colored in",
			default=(0.0, 0.7, 0.0, 1.0),
			min=0.0, soft_max=1.0,
			size=4,
			subtype='COLOR'
			)
	spine_z_color: FloatVectorProperty(name="Z color",
			description="Color that spines corresponding to Z rotation will be colored in",
			default=(0.0, 0.0, 0.7, 1.0),
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
	highlight_size: FloatProperty(name="Highlight size",
			description="Size in pixels that the highlight circle extends beyond the highlighted item",
			default=5.0,
			min=0.0,
			step=1.0
			)


	highlight_do_outline: BoolProperty(name="Highlight outline",
			description="Whether to draw outlines for the highlight circle",
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
			
configurable_props = ["use_depsgraph", "allow_negative_scale", #"allow_negative_handle_scale",
["do_location", "do_rotation", "do_scale"], "retime_old_y",
"select_key", "select_threshold", "deselect_nohit_key", "deselect_always_key", "deselect_passthrough", "mode", 
"path_style", "simple_color", "speed_color_min", "speed_color_max", "accel_color_neg", "accel_color_static", "accel_color_pos",
"keyframe_color", "selection_color", "selection_color_dark", 
"highlight_color", "highlight_size", "highlight_do_outline",
["point_color_loc", "point_color_rot", "point_color_scl"], "point_outline_size", "point_outline_blur",
"keyframe_size", "frame_size", "frame_color",
"handle_color_fac", "handle_line_color", "handle_size", 
"timebead_size", "timebead_color", 
["sensitivity_location", "sensitivity_rotation", "sensitivity_scale"], "sensitivity_shift", "sensitivity_alt",
"text_color", "selected_text_color", "keyframe_text_size", "keyframe_text_offset_x", "keyframe_text_offset_y",
"path_width", "path_outline_width", "path_step", "path_before", "path_after", "pretty_lines",
"keyframe_numbers", "frame_display", 
"handle_display", "handle_length", "handle_size", "handle_direction", 
"show_spines", "spine_length", "spine_step", "spine_offset",
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
		#MotionTrailSaveUserpref,
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
