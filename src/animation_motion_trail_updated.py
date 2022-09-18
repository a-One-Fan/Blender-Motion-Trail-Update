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
	"version": (0, 13, 1),
	"blender": (3, 2, 0),
	"location": "View3D > Toolbar > Motion Trail tab",
	"warning": "Support for features not originally present is buggy; NO UNDO!!!",
	"description": "Display and edit motion trails in the 3D View",
	"category": "Animation",
}


import bgl
import gpu
from gpu_extras.batch import batch_for_shader
import blf
import bpy
import bpy_types
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

# Linear interpolation for 4-element tuples
def lerp4(fac, tup1, tup2):
	return (* [tup1[i] * fac + tup2[i]*(1.0-fac) for i in range(4)],)

# fake fcurve class, used if no fcurve is found for a path
class fake_fcurve():
	def __init__(self, object, index, rotation=False, scale=False):
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


# get location curves of the given object
def get_curves(object, child=False):
	if object.animation_data and object.animation_data.action:
		action = object.animation_data.action
		if child:
			# posebone
			curves = [
					fc for fc in action.fcurves if len(fc.data_path) >= 14 and
					fc.data_path[-9:] == '.location' and
					child.name in fc.data_path.split("\"")
					]
		else:
			# normal object
			curves = [fc for fc in action.fcurves if fc.data_path == 'location']

	elif object.animation_data and object.animation_data.use_nla:
		curves = []
		strips = []
		for track in object.animation_data.nla_tracks:
			not_handled = [s for s in track.strips]
			while not_handled:
				current_strip = not_handled.pop(-1)
				if current_strip.action:
					strips.append(current_strip)
				if current_strip.strips:
					# meta strip
					not_handled += [s for s in current_strip.strips]

		for strip in strips:
			if child:
				# posebone
				curves = [
						fc for fc in strip.action.fcurves if
						len(fc.data_path) >= 14 and fc.data_path[-9:] == '.location' and
						child.name in fc.data_path.split("\"")
						]
			else:
				# normal object
				curves = [fc for fc in strip.action.fcurves if fc.data_path == 'location']
			if curves:
				# use first strip with location fcurves
				break
	else:
		# should not happen?
		curves = []

	# ensure we have three curves per object
	fcx = None
	fcy = None
	fcz = None
	for fc in curves:
		if fc.array_index == 0:
			fcx = fc
		elif fc.array_index == 1:
			fcy = fc
		elif fc.array_index == 2:
			fcz = fc
	if fcx is None:
		fcx = fake_fcurve(object, 0)
	if fcy is None:
		fcy = fake_fcurve(object, 1)
	if fcz is None:
		fcz = fake_fcurve(object, 2)

	return([fcx, fcy, fcz])


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
		mathutils.Vector((vector[0], vector[1], vector[2], 1.0))
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

# Turn location, rotation and scale in an action into a usable matrix
def get_matrix_action(action, frame, locpath, rotpath, sclpath, quat,
defaultLoc = None, defaultRot = None, defaultScale = None):
	rotrange = 4
	if not quat:
		rotrange = 3
	
	loccurves = [action.fcurves.find(locpath, index=i) for i in range(3)]
	rotcurves = [action.fcurves.find(rotpath, index=i) for i in range(rotrange)]
	sclcurves = [action.fcurves.find(sclpath, index=i) for i in range(3)]
	
	loc = defaultLoc
	if loccurves[0] is not None:
		loc = mathutils.Vector([c.evaluate(frame) for c in loccurves])
	
	rot = defaultRot
	if rotcurves[0] is not None:
		if rotrange == 4:
			rot = mathutils.Quaternion([c.evaluate(frame) for c in rotcurves])
		else:
			rot = mathutils.Euler([c.evaluate(frame) for c in rotcurves])
			
	scale = defaultScale
	if sclcurves[0] is not None:
		scale = mathutils.Vector([c.evaluate(frame) for c in sclcurves])
	
	return mathutils.Matrix.LocRotScale(loc, rot, scale)

# Get the locrotscale matrix from the fcurves for a given frame and action for an object
# (or posebone, for which this works fine as well)
def get_matrix_frame(obj, frame, action):
	locpath = obj.path_from_id("location")
	rotpath = ""
	
	quat = True
	if obj.rotation_mode == 'QUATERNION':
		rotpath = obj.path_from_id("rotation_quaternion")
	else:
		rotpath = obj.path_from_id('rotation_euler')
		quat = False
	sclpath = obj.path_from_id("scale")
	
	return get_matrix_action(action, frame, locpath, rotpath, sclpath, quat,
	obj.location, obj.rotation_quaternion if quat else obj.rotation_euler, obj.scale)
	
# Get the world-ish matrix for an object, factoring in its parents recursively, if any
def get_matrix_obj_parents(obj, frame, do_anim=True):
	mat = None
	
	if do_anim:
		mat = get_matrix_frame(obj, frame, obj.animation_data.action)
	else:
		mat = mathutils.Matrix()

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
		animMat = mathutils.Matrix()
		
	parentMat = None
	parentOffsetMat = None
	if pose_bone.parent:
		parentMat = get_matrix_bone_parents_as(pose_bone.parent, frame)
		parentOffsetMat = pose_bone.parent.bone.matrix_local.inverted() @ pose_bone.bone.matrix_local
	else:
		parentMat = mathutils.Matrix()
		parentOffsetMat = pose_bone.bone.matrix_local
		
	res = parentMat @ parentOffsetMat @ animMat
		
	if pose_bone.constraints:
		res = evaluate_constraints(res, pose_bone.constraints, frame, pose_bone)
	
	return res

def get_matrix_bone_parents(pose_bone, frame, do_anim = True):
	return get_matrix_obj_parents(pose_bone.id_data, frame) @ \
	get_matrix_bone_parents_as(pose_bone, frame, do_anim)

# Get the world-ish matrix of a bone or object
def get_matrix_any_parents(thing, frame, do_anim = True):
	if type(thing) is bpy.types.PoseBone:
		return get_matrix_bone_parents(thing, frame, do_anim)
	return get_matrix_obj_parents(thing, frame, do_anim)

# Get matrix for child of constraint
def evaluate_childof(constraint, frame):
	mat = mathutils.Matrix()
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
			mat = mathutils.Matrix.LocRotScale(disassembledLoc, disassembledRot, disassembledScl)
	
	except Exception as e:
		print(e)
		tb = sys.exc_info()[-1]
		print(traceback.extract_tb(tb))
	
	finally:
		return mat
			
constraint_funcs = {'CHILD_OF': evaluate_childof}

# Get matrices from all constraints?
def evaluate_constraints(mat, constraints, frame, ob):
	accumulatedMat = mathutils.Matrix()
	for c in constraints:
		f = constraint_funcs.get(c.type)
		if f is None or not c.enabled or c.influence == 0.0:
			continue
		constraintMat = f(c, frame)
		if c.influence != 1.0:
			constraintMat = constraintMat.lerp(mathutils.Matrix(), 1.0-c.influence)
		accumulatedMat = accumulatedMat @ constraintMat
	return accumulatedMat @ mat

# calculate location of display_ob in worldspace using our own, draw handler-safe methods
def get_location(frame, display_ob, offset_ob, curves, context):
	return (get_matrix_any_parents(display_ob, frame).to_translation())

# calculate location of display_ob in worldspace using the depsgraph
def get_location_depsgraph(frame, display_ob, context):
	ct_otherframe = context.copy()
	ct_otherframe["frame"] = frame
	
	dg = ct_otherframe.evaluated_depsgraph_get()
	boneMat = mathutils.Matrix()
	ob = display_ob
	
	if type(display_ob) is bpy.types.PoseBone:
		evalledBone = display_ob.evaluated_get(dg)
		boneMat = evalledBone.matrix
		ob = display_ob.id_data
		
	return (ob.evaluated_get(dg).matrix_world @ boneMat).location()

# Calculate an inverse matrix for an object or bone, such that it's suitable for the addon's
# manipulation of keyframes (IE without the very last animation applied)
# using our own, draw handler-safe methods
def get_inverse_parents(frame, ob, context):
	return get_matrix_any_parents(ob, frame, False).inverted()

def get_inverse_parents_depsgraph(frame, ob, context):
	mat = ''
	if type(ob) is bpy.types.PoseBone:
		mat = get_matrix_frame(ob, frame, ob.id_data.animation_data.action)
	else:
		mat = get_matrix_frame(ob, frame, ob.animation_data.action)
		
	return (get_location_depsgraph(frame, ob, context) @ mat.inverted()).inverted()

# get position of keyframes and handles at the start of dragging
def get_original_animation_data(context, keyframes, location_getter):
	keyframes_ori = {}
	handles_ori = {}

	if context.active_object and context.active_object.mode == 'POSE':
		armature_ob = context.active_object
		objects = [[armature_ob, pb, armature_ob] for pb in
					context.selected_pose_bones]
	else:
		objects = [[ob, False, False] for ob in context.selected_objects]

	for action_ob, child, offset_ob in objects:
		if not action_ob.animation_data:
			continue
		curves = get_curves(action_ob, child)
		if len(curves) == 0:
			continue
		fcx, fcy, fcz = curves
		if child:
			display_ob = child
		else:
			display_ob = action_ob

		# get keyframe positions
		frame_old = context.scene.frame_current
		keyframes_ori[display_ob.name] = {}
		for frame in keyframes[display_ob.name]:
			loc = location_getter(frame, display_ob, offset_ob, curves, context)
			keyframes_ori[display_ob.name][frame] = [frame, loc]

		# get handle positions
		handles_ori[display_ob.name] = {}
		for frame in keyframes[display_ob.name]:
			handles_ori[display_ob.name][frame] = {}
			left_x = [frame, fcx.evaluate(frame)]
			right_x = [frame, fcx.evaluate(frame)]
			for kf in fcx.keyframe_points:
				if kf.co[0] == frame:
					left_x = kf.handle_left[:]
					right_x = kf.handle_right[:]
					break
			left_y = [frame, fcy.evaluate(frame)]
			right_y = [frame, fcy.evaluate(frame)]
			for kf in fcy.keyframe_points:
				if kf.co[0] == frame:
					left_y = kf.handle_left[:]
					right_y = kf.handle_right[:]
					break
			left_z = [frame, fcz.evaluate(frame)]
			right_z = [frame, fcz.evaluate(frame)]
			for kf in fcz.keyframe_points:
				if kf.co[0] == frame:
					left_z = kf.handle_left[:]
					right_z = kf.handle_right[:]
					break
			handles_ori[display_ob.name][frame]["left"] = [left_x, left_y,
				left_z]
			handles_ori[display_ob.name][frame]["right"] = [right_x, right_y,
				right_z]

		if context.scene.frame_current != frame_old:
			context.scene.frame_set(frame_old)

	return(keyframes_ori, handles_ori)

def get_original_animation_data_dg(context, keyframes):
	return get_original_animation_data(context, keyframes, get_location_depsgraph)

def get_original_animation_data_ce(context, keyframes):
	return get_original_animation_data(context, keyframes, get_location)

# callback function that calculates positions of all things that need be drawn
def calc_callback(self, context, inverse_getter, location_getter):
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
	
	if context.active_object and context.active_object.mode == 'POSE':
		armature_ob = context.active_object
		objects = [
				[armature_ob, pb, armature_ob] for pb in
				context.selected_pose_bones
				]
	else:
		objects = [[ob, False, False] for ob in context.selected_objects]
	if objects == self.displayed:
		selection_change = False
	else:
		selection_change = True

	if self.lock and not selection_change and \
	context.region_data.perspective_matrix == self.perspective and not \
	context.window_manager.motion_trail.force_update:
		pass
		#return
	# dictionaries with key: objectname
	self.paths = {} 	 # value: list of lists with x, y, color
	self.keyframes = {}  # value: dict with frame as key and [x,y] as value
	self.handles = {}    # value: dict of dicts
	self.timebeads = {}  # value: dict with frame as key and [x,y] as value
	self.click = {} 	 # value: list of lists with frame, type, loc-vector
	if selection_change:
		# value: editbone inverted rotation matrix or None
		self.active_keyframe = False
		self.active_handle = False
		self.active_timebead = False
		self.active_frame = False
	if selection_change or not self.lock or context.window_manager.\
	motion_trail.force_update:
		# contains locations of path, keyframes and timebeads
		self.cached = {
				"path": {}, "keyframes": {}, "timebeads_timing": {},
				"timebeads_speed": {}
				}
	if self.cached["path"]:
		use_cache = True
	else:
		use_cache = False
	self.perspective = context.region_data.perspective_matrix.copy()
	self.displayed = objects  # store, so it can be checked next time
	context.window_manager.motion_trail.force_update = False
	try:
		#global_undo = context.preferences.edit.use_global_undo
		#context.preferences.edit.use_global_undo = False
		for action_ob, child, offset_ob in objects:
			if not action_ob.animation_data:
				continue
			curves = get_curves(action_ob, child)
			if len(curves) == 0:
				continue
			if context.window_manager.motion_trail.path_before == 0:
				range_min = context.scene.frame_start
			else:
				range_min = max(
							context.scene.frame_start,
							context.scene.frame_current -
							context.window_manager.motion_trail.path_before
							)
			if context.window_manager.motion_trail.path_after == 0:
				range_max = context.scene.frame_end
			else:
				range_max = min(context.scene.frame_end,
							context.scene.frame_current +
							context.window_manager.motion_trail.path_after
							)
			fcx, fcy, fcz = curves
			if child:
				display_ob = child
			else:
				display_ob = action_ob
			# get location data of motion path
			path = []
			speeds = []
			frame_old = context.scene.frame_current
			step = 11 - context.window_manager.motion_trail.path_resolution
			if not use_cache:
				if display_ob.name not in self.cached["path"]:
					self.cached["path"][display_ob.name] = {}
					
			if use_cache and range_min - 1 in self.cached["path"][display_ob.name]:
				prev_loc = self.cached["path"][display_ob.name][range_min - 1]
			else:
				prev_loc = location_getter(range_min - 1, display_ob, offset_ob, curves, context)
				self.cached["path"][display_ob.name][range_min - 1] = prev_loc
			for frame in range(range_min, range_max + 1, step):
				if use_cache and frame in self.cached["path"][display_ob.name]:
					loc = self.cached["path"][display_ob.name][frame]
				else:
					loc = location_getter(frame, display_ob, offset_ob, curves, context)
					self.cached["path"][display_ob.name][frame] = loc
				if not context.region or not context.space_data:
					continue
				x, y = world_to_screen(context, loc)
				if context.window_manager.motion_trail.path_style == 'simple':
					path.append([x, y, [0.0, 0.0, 0.0], frame, action_ob, child])
				else:
					dloc = (loc - prev_loc).length
					path.append([x, y, dloc, frame, action_ob, child])
					speeds.append(dloc)
					prev_loc = loc
			# calculate color of path
			if context.window_manager.motion_trail.path_style == 'speed':
				speeds.sort()
				min_speed = speeds[0]
				d_speed = speeds[-1] - min_speed
				d_speed = max(d_speed, 1e-6)
				for i, [x, y, d_loc, frame, action_ob, child] in enumerate(path):
					relative_speed = (d_loc - min_speed) / d_speed # 0.0 to 1.0
					fac = min(1.0, 2.0 * relative_speed)
					path[i][2] = lerp4(fac, context.window_manager.motion_trail.speed_color_max, 
					context.window_manager.motion_trail.speed_color_min)
			elif context.window_manager.motion_trail.path_style == 'acceleration':
				accelerations = []
				prev_speed = 0.0
				for i, [x, y, d_loc, frame, action_ob, child] in enumerate(path):
					accel = d_loc - prev_speed
					accelerations.append(accel)
					path[i][2] = accel
					prev_speed = d_loc
				accelerations.sort()
				min_accel = accelerations[0]
				max_accel = accelerations[-1]
				for i, [x, y, accel, frame, action_ob, child] in enumerate(path):
					if accel < 0:
						relative_accel = accel / min_accel  # values from 0.0 to 1.0
						fac = 1.0 - relative_accel
						path[i][2] = lerp4(fac, context.window_manager.motion_trail.accel_color_neg, 
						context.window_manager.motion_trail.accel_color_static)
					elif accel > 0:
						relative_accel = accel / max_accel  # values from 1.0 to 0.0
						fac = 1.0 - relative_accel
						path[i][2] = lerp4(fac, context.window_manager.motion_trail.accel_color_static, 
						context.window_manager.motion_trail.accel_color_pos)
					else:
						path[i][2] = context.window_manager.motion_trail.accel_color_static
			self.paths[display_ob.name] = path
			# get keyframes and handles
			keyframes = {}
			handle_difs = {}
			kf_time = []
			click = []
			if not use_cache:
				if display_ob.name not in self.cached["keyframes"]:
					self.cached["keyframes"][display_ob.name] = {}

			for fc in curves:
				for kf in fc.keyframe_points:
					# handles for location mode
					if context.window_manager.motion_trail.mode == 'location':
						if kf.co[0] not in handle_difs:
							handle_difs[kf.co[0]] = {"left": mathutils.Vector(),
								"right": mathutils.Vector(), "keyframe_loc": None}
								
						ldiff = mathutils.Vector(kf.handle_left[:]) - mathutils.Vector(kf.co[:])
						rdiff = mathutils.Vector(kf.handle_right[:]) - mathutils.Vector(kf.co[:])
						hdir = context.window_manager.motion_trail.handle_direction
						lco = 0.0
						rco = 0.0
						
						if hdir == 'time':
							lco = ldiff.normalized()[1]
							rco = rdiff.normalized()[1]
						elif hdir == 'wtime':
							lco = sum(ldiff.normalized() * mathutils.Vector((0.25, 0.75)))
							rco = sum(rdiff.normalized() * mathutils.Vector((0.25, 0.75)))
						elif hdir == 'location':
							lco = ldiff.normalized()[0]
							rco = rdiff.normalized()[0]
						elif hdir == 'wloc':
							lco = sum(ldiff.normalized() * mathutils.Vector((0.75, 0.25)))
							rco = sum(rdiff.normalized() * mathutils.Vector((0.75, 0.25)))
						elif hdir == 'len':
							lco = -ldiff.length
							rco = rdiff.length
						
						handle_difs[kf.co[0]]["left"][fc.array_index] = lco
						handle_difs[kf.co[0]]["right"][fc.array_index] = rco
					# keyframes
					if kf.co[0] in kf_time:
						continue
					kf_time.append(kf.co[0])
					co = kf.co[0]

					if use_cache and co in \
					self.cached["keyframes"][display_ob.name]:
						loc = self.cached["keyframes"][display_ob.name][co]
					else:
						loc = location_getter(co, display_ob, offset_ob, curves, context)
						self.cached["keyframes"][display_ob.name][co] = loc
					if handle_difs:
						handle_difs[co]["keyframe_loc"] = loc

					x, y = world_to_screen(context, loc)
					keyframes[kf.co[0]] = [x, y]
					if context.window_manager.motion_trail.mode != 'speed':
						# can't select keyframes in speed mode
						click.append([kf.co[0], "keyframe",
							mathutils.Vector([x, y]), action_ob, child])
			self.keyframes[display_ob.name] = keyframes
			# handles are only shown in location-altering mode
			if context.window_manager.motion_trail.mode == 'location' and \
			context.window_manager.motion_trail.handle_display:
				# calculate handle positions
				handles = {}
				for frame, vecs in handle_difs.items():
					if child:
						# bone space to world space
						mat = inverse_getter(frame, child, context)
						vec_left = vecs["left"] @ mat
						vec_right = vecs["right"] @ mat
					else:
						vec_left = vecs["left"]
						vec_right = vecs["right"]
						
					hlen = context.window_manager.motion_trail.handle_length
					vec_left = vec_left * hlen
					vec_right = vec_right * hlen
					if vecs["keyframe_loc"] is not None:
						vec_keyframe = vecs["keyframe_loc"]
					else:
						vec_keyframe = location_getter(frame, display_ob, offset_ob,
							curves, context)
					x_left, y_left = world_to_screen(
											context, vec_left * 2 + vec_keyframe
											)
					x_right, y_right = world_to_screen(
											context, vec_right * 2 + vec_keyframe
											)
					handles[frame] = {"left": [x_left, y_left],
									"right": [x_right, y_right]}
					click.append([frame, "handle_left",
						mathutils.Vector([x_left, y_left]), action_ob, child])
					click.append([frame, "handle_right",
						mathutils.Vector([x_right, y_right]), action_ob, child])
				self.handles[display_ob.name] = handles

			# calculate timebeads for timing mode
			if context.window_manager.motion_trail.mode == 'timing':
				timebeads = {}
				n = context.window_manager.motion_trail.timebeads * (len(kf_time) - 1)
				dframe = (range_max - range_min) / (n + 1)
				if not use_cache:
					if display_ob.name not in self.cached["timebeads_timing"]:
						self.cached["timebeads_timing"][display_ob.name] = {}

				for i in range(1, n + 1):
					frame = range_min + i * dframe
					if use_cache and frame in \
							self.cached["timebeads_timing"][display_ob.name]:
						loc = self.cached["timebeads_timing"][display_ob.name][frame]
					else:
						loc = location_getter(frame, display_ob, offset_ob, curves, context)
						self.cached["timebeads_timing"][display_ob.name][frame] = loc
					x, y = world_to_screen(context, loc)
					timebeads[frame] = [x, y]
					click.append(
							[frame, "timebead", mathutils.Vector([x, y]),
							action_ob, child]
							)
				self.timebeads[display_ob.name] = timebeads

			# calculate timebeads for speed mode
			if context.window_manager.motion_trail.mode == 'speed':
				angles = dict([[kf, {"left": [], "right": []}] for kf in
							  self.keyframes[display_ob.name]])
				for fc in curves:
					for i, kf in enumerate(fc.keyframe_points):
						if i != 0:
							angle = mathutils.Vector([-1, 0]).angle(
												mathutils.Vector(kf.handle_left) -
												mathutils.Vector(kf.co), 0
												)
							if angle != 0:
								angles[kf.co[0]]["left"].append(angle)
						if i != len(fc.keyframe_points) - 1:
							angle = mathutils.Vector([1, 0]).angle(
												mathutils.Vector(kf.handle_right) -
												mathutils.Vector(kf.co), 0
												)
							if angle != 0:
								angles[kf.co[0]]["right"].append(angle)
				timebeads = {}
				kf_time.sort()
				if not use_cache:
					if display_ob.name not in self.cached["timebeads_speed"]:
						self.cached["timebeads_speed"][display_ob.name] = {}

				for frame, sides in angles.items():
					if sides["left"]:
						perc = (sum(sides["left"]) / len(sides["left"])) / \
							(math.pi / 2)
						perc = max(0.4, min(1, perc * 5))
						previous = kf_time[kf_time.index(frame) - 1]
						bead_frame = frame - perc * ((frame - previous - 2) / 2)
						if use_cache and bead_frame in \
						self.cached["timebeads_speed"][display_ob.name]:
							loc = self.cached["timebeads_speed"][display_ob.name][bead_frame]
						else:
							loc = location_getter(bead_frame, display_ob, offset_ob,
								curves, context)
							self.cached["timebeads_speed"][display_ob.name][bead_frame] = loc
						x, y = world_to_screen(context, loc)
						timebeads[bead_frame] = [x, y]
						click.append(
								[bead_frame, "timebead",
								mathutils.Vector([x, y]),
								action_ob, child]
								)
					if sides["right"]:
						perc = (sum(sides["right"]) / len(sides["right"])) / \
							(math.pi / 2)
						perc = max(0.4, min(1, perc * 5))
						next = kf_time[kf_time.index(frame) + 1]
						bead_frame = frame + perc * ((next - frame - 2) / 2)
						if use_cache and bead_frame in \
						self.cached["timebeads_speed"][display_ob.name]:
							loc = self.cached["timebeads_speed"][display_ob.name][bead_frame]
						else:
							loc = location_getter(bead_frame, display_ob, offset_ob,
								curves, context)
							self.cached["timebeads_speed"][display_ob.name][bead_frame] = loc
						x, y = world_to_screen(context, loc)
						timebeads[bead_frame] = [x, y]
						click.append(
								[bead_frame, "timebead",
								mathutils.Vector([x, y]),
								action_ob, child]
								)
				self.timebeads[display_ob.name] = timebeads
			# add frame positions to click-list
			if context.window_manager.motion_trail.frame_display:
				path = self.paths[display_ob.name]
				for x, y, color, frame, action_ob, child in path:
					click.append(
							[frame, "frame",
							mathutils.Vector([x, y]),
							action_ob, child]
							)

			self.click[display_ob.name] = click

			if context.scene.frame_current != frame_old:
				context.scene.frame_set(frame_old)

		#context.preferences.edit.use_global_undo = global_undo

	except Exception as e:
		print(e)
		tb = sys.exc_info()[-1]
		print(traceback.extract_tb(tb))
		# restore global undo in case of failure (see T52524)
		#context.preferences.edit.use_global_undo = global_undo

# calc_callback using depsgraph functions
def calc_callback_dg(self, context):
	return calc_callback(self, context, get_inverse_parents_depsgraph, get_location_depsgraph)

# calc_callback using custom evaluation functions
def calc_callback_ce(self, context):
	return calc_callback(self, context, get_inverse_parents, get_location_depsgraph)


# draw in 3d-view
def draw_callback(self, context):
	# Remove handler if file was changed and we lose access to self
	try:
		self.properties
	except:
		if global_mtrail_handler_draw:
			bpy.types.SpaceView3D.draw_handler_remove(global_mtrail_handler_draw, 'WINDOW')
		return
	
	# polling
	if (context.mode not in ('OBJECT', 'POSE') or
			not context.window_manager.motion_trail.enabled):
		return

	# display limits
	if context.window_manager.motion_trail.path_before != 0:
		limit_min = context.scene.frame_current - \
			context.window_manager.motion_trail.path_before
	else:
		limit_min = -1e6
	if context.window_manager.motion_trail.path_after != 0:
		limit_max = context.scene.frame_current + \
			context.window_manager.motion_trail.path_after
	else:
		limit_max = 1e6
	# draw motion path
	width = context.window_manager.motion_trail.path_width
	#uniform_line_shader = gpu.shader.from_builtin('3D_POLYLINE_UNIFORM_COLOR')
	colored_line_shader = gpu.shader.from_builtin('3D_POLYLINE_SMOOTH_COLOR')
	colored_points_shader = gpu.shader.from_builtin('2D_FLAT_COLOR')
	
	poss = []
	cols = []
	
	if context.window_manager.motion_trail.path_style == 'simple':
		#uniform_line_shader.bind()
		#uniform_line_shader.uniform_float("color", context.window_manager.motion_trail.simple_color)
		#uniform_line_shader.uniform_float("lineWidth", width)
		
		colored_line_shader.bind()
		colored_line_shader.uniform_float("lineWidth", width)
		simple_color = context.window_manager.motion_trail.simple_color
		for objectname, path in self.paths.items():
			for x, y, color, frame, action_ob, child in path:
				if frame < limit_min or frame > limit_max:
					continue
				poss.append((x, y, 0))
				cols.append(simple_color)
			#batch = batch_for_shader(uniform_line_shader, 'LINE_STRIP', {"pos": poss})
			batch = batch_for_shader(colored_line_shader, 'LINE_STRIP', {"pos": poss, "color": cols})
			batch.draw(colored_line_shader)
			poss.clear()
			cols.clear()
	else:
		colored_line_shader.bind()
		colored_line_shader.uniform_float("lineWidth", width)
		for objectname, path in self.paths.items():
			for i, [x, y, color, frame, action_ob, child] in enumerate(path):
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
			
			batch = batch_for_shader(colored_line_shader, 'LINE_STRIP', {"pos": poss, "color": cols})
			batch.draw(colored_line_shader)
			poss.clear()
			cols.clear()

	# draw frames
	if context.window_manager.motion_trail.frame_display:
		colored_points_shader.bind()
		point_poss = []
		point_cols = []
		for objectname, path in self.paths.items():
			for x, y, color, frame, action_ob, child in path:
				if frame < limit_min or frame > limit_max:
					continue
				if self.active_frame and objectname == self.active_frame[0] \
				and abs(frame - self.active_frame[1]) < 1e-4:
					point_cols.append(context.window_manager.motion_trail.selection_color)
					point_poss.append((x, y))
				else:
					point_poss.append((x, y))
					point_cols.append(context.window_manager.motion_trail.frame_color)
			batch = batch_for_shader(colored_points_shader, 'POINTS', {"pos": point_poss, "color": point_cols})
			gpu.state.point_size_set(3.0)
			batch.draw(colored_points_shader)
			point_poss.clear()
			point_cols.clear()

	# time beads are shown in speed and timing modes
	if context.window_manager.motion_trail.mode in ('speed', 'timing'):
		gpu.state.point_size_set(4.0)
		point_poss = []
		point_cols = []
		for objectname, values in self.timebeads.items():
			for frame, coords in values.items():
				if frame < limit_min or frame > limit_max:
					continue
				if self.active_timebead and \
				objectname == self.active_timebead[0] and \
				abs(frame - self.active_timebead[1]) < 1e-4:
					point_cols.append(context.window_manager.motion_trail.selection_color)
					point_poss.append((coords[0], coords[1]))
				else:
					point_cols.append(context.window_manager.motion_trail.timebead_color)
					point_poss.append((coords[0], coords[1]))
			batch = batch_for_shader(colored_points_shader, 'POINTS', {"pos": point_poss, "color": point_cols})
			gpu.state.point_size_set(3.0)
			batch.draw(colored_points_shader)
			point_poss.clear()
			point_cols.clear()

	# handles are only shown in location mode
	if context.window_manager.motion_trail.mode == 'location':
		colored_line_shader.bind()
		colored_line_shader.uniform_float("lineWidth", 2)
		poss = []
		cols = []
		for objectname, values in self.handles.items():
			for frame, sides in values.items():
				if frame < limit_min or frame > limit_max:
					continue
				for side, coords in sides.items():
					if self.active_handle and \
					objectname == self.active_handle[0] and \
					side == self.active_handle[2] and \
					abs(frame - self.active_handle[1]) < 1e-4:
						cols.append(context.window_manager.motion_trail.selection_color_dark)
						poss.append((self.keyframes[objectname][frame][0],
							self.keyframes[objectname][frame][1], 0.0))
						cols.append(context.window_manager.motion_trail.selection_color_dark)
						poss.append((coords[0], coords[1], 0.0))
						
					else:
						cols.append(context.window_manager.motion_trail.handle_line_color)
						poss.append((self.keyframes[objectname][frame][0],
							self.keyframes[objectname][frame][1], 0.0))
						cols.append(context.window_manager.motion_trail.handle_line_color)
						poss.append((coords[0], coords[1], 0.0))
			batch = batch_for_shader(colored_line_shader, 'LINES', {"pos": poss, "color": cols})
			batch.draw(colored_line_shader)
			poss.clear()
			cols.clear()

		# draw handles
		colored_points_shader.bind()
		gpu.state.point_size_set(4.0)
		point_poss = []
		point_cols = []
		for objectname, values in self.handles.items():
			for frame, sides in values.items():
				if frame < limit_min or frame > limit_max:
					continue
				for side, coords in sides.items():
					if self.active_handle and \
					objectname == self.active_handle[0] and \
					side == self.active_handle[2] and \
					abs(frame - self.active_handle[1]) < 1e-4:
						point_poss.append((coords[0], coords[1]))
						point_cols.append(context.window_manager.motion_trail.selection_color)
					else:
						point_poss.append((coords[0], coords[1]))
						point_cols.append(context.window_manager.motion_trail.handle_color)
		batch = batch_for_shader(colored_points_shader, 'POINTS', {"pos": point_poss, "color": point_cols})
		batch.draw(colored_points_shader)
		point_poss.clear()
		point_cols.clear()

	# draw keyframes
	colored_points_shader.bind()
	gpu.state.point_size_set(6.0)
	point_poss = []
	point_cols = []
	for objectname, values in self.keyframes.items():
		for frame, coords in values.items():
			if frame < limit_min or frame > limit_max:
				continue
			if self.active_keyframe and \
			objectname == self.active_keyframe[0] and \
			abs(frame - self.active_keyframe[1]) < 1e-4:
				point_poss.append((coords[0], coords[1]))
				point_cols.append(context.window_manager.motion_trail.selection_color)
			else:
				point_poss.append((coords[0], coords[1]))
				point_cols.append(context.window_manager.motion_trail.handle_color)
	batch = batch_for_shader(colored_points_shader, 'POINTS', {"pos": point_poss, "color": point_cols})
	batch.draw(colored_points_shader)
	point_poss.clear()
	point_cols.clear()

	# draw keyframe-numbers
	if context.window_manager.motion_trail.keyframe_numbers:
		blf.size(0, 12, 72)
		blf.color(0, 1.0, 1.0, 0.0, 1.0)
		for objectname, values in self.keyframes.items():
			for frame, coords in values.items():
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
				objectname == self.active_keyframe[0] and \
				abs(frame - self.active_keyframe[1]) < 1e-4:
					c = context.window_manager.motion_trail.selected_text_color
					blf.color(0, * c)
					blf.draw(0, text)
				else:
					c = context.window_manager.motion_trail.text_color
					blf.color(0, * c)
					blf.draw(0, text)

	# restore opengl defaults
	gpu.state.point_size_set(1.0) # TODO: is this the correct value?

# change data based on mouse movement
def drag(context, event, drag_mouse_ori, active_keyframe, active_handle,
active_timebead, keyframes_ori, handles_ori, inverse_getter):
	# change 3d-location of keyframe
	if context.window_manager.motion_trail.mode == 'location' and \
	active_keyframe:
		objectname, frame, frame_ori, action_ob, child = active_keyframe
		mat = inverse_getter(frame, child if child else action_ob, context)
		mt = context.window_manager.motion_trail
		
		mouse_ori_world = mat @ screen_to_world(context, drag_mouse_ori[0],
			drag_mouse_ori[1])
		vector = mat @ screen_to_world(context, event.mouse_region_x,
			event.mouse_region_y)
		d = vector - mouse_ori_world

		loc_ori_ws = keyframes_ori[objectname][frame][1]
		loc_ori_ls = mat @ loc_ori_ws
		new_loc = loc_ori_ls + d
		curves = get_curves(action_ob, child)
		
		if not mt.backed_up_keyframes:
			for i, curve in enumerate(curves):
				for kf in curve.keyframe_points:
					if kf.co[0] == frame:
						mt.keyframe_backup[i] = kf.co[1]
						break
			mt.backed_up_keyframes = True
		
		for i, curve in enumerate(curves):
			for kf in curve.keyframe_points:
				if kf.co[0] == frame:
					kf.co[1] = new_loc[i]
					kf.handle_left[1] = handles_ori[objectname][frame]["left"][i][1] + d[i]
					kf.handle_right[1] = handles_ori[objectname][frame]["right"][i][1] + d[i]
					break

	# change 3d-location of handle
	elif context.window_manager.motion_trail.mode == 'location' and active_handle:
		objectname, frame, side, action_ob, child = active_handle
		mat = inverse_getter(frame, child if child else action_ob, context)

		mouse_ori_world = mat @ screen_to_world(context, drag_mouse_ori[0],
			drag_mouse_ori[1])
		vector = mat @ screen_to_world(context, event.mouse_region_x,
			event.mouse_region_y)
		d = vector - mouse_ori_world
		curves = get_curves(action_ob, child)

		for i, curve in enumerate(curves):
			for kf in curve.keyframe_points:
				if kf.co[0] == frame:
					if side == "left":
						# change handle type, if necessary
						if kf.handle_left_type in (
								'AUTO',
								'AUTO_CLAMPED',
								'ANIM_CLAMPED'):
							kf.handle_left_type = 'ALIGNED'
						elif kf.handle_left_type == 'VECTOR':
							kf.handle_left_type = 'FREE'
						# change handle position(s)
						kf.handle_left[1] = handles_ori[objectname][frame]["left"][i][1] + d[i]
						if kf.handle_left_type in (
								'ALIGNED',
								'ANIM_CLAMPED',
								'AUTO',
								'AUTO_CLAMPED'):
							dif = (
								abs(handles_ori[objectname][frame]["right"][i][0] -
								kf.co[0]) / abs(kf.handle_left[0] -
								kf.co[0])
								) * d[i]
							kf.handle_right[1] = handles_ori[objectname][frame]["right"][i][1] - dif
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
						kf.handle_right[1] = handles_ori[objectname][frame]["right"][i][1] + d[i]
						if kf.handle_right_type in (
								'ALIGNED',
								'ANIM_CLAMPED',
								'AUTO',
								'AUTO_CLAMPED'):
							dif = (
								abs(handles_ori[objectname][frame]["left"][i][0] -
								kf.co[0]) / abs(kf.handle_right[0] -
								kf.co[0])
								) * d[i]
							kf.handle_left[1] = handles_ori[objectname][frame]["left"][i][1] - dif
					break

	# change position of all keyframes on timeline
	elif context.window_manager.motion_trail.mode == 'timing' and \
	active_timebead:
		objectname, frame, frame_ori, action_ob, child = active_timebead
		curves = get_curves(action_ob, child)
		ranges = [val for c in curves for val in c.range()]
		ranges.sort()
		range_min = round(ranges[0])
		range_max = round(ranges[-1])
		range = range_max - range_min
		dx_screen = -(mathutils.Vector([event.mouse_region_x,
			event.mouse_region_y]) - drag_mouse_ori)[0]
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
				for f in keyframes_ori[objectname]:
					if abs(f - frame_map) < 1e-4:
						frame_ori = keyframes_ori[objectname][f][0]
						value_ori = keyframes_ori[objectname][f]
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
				kf.handle_left[0] = handles_ori[objectname][frame_ori]["left"][i][0] + d_frame
				kf.handle_right[0] = handles_ori[objectname][frame_ori]["right"][i][0] + d_frame
		del keyframes_ori[objectname]
		keyframes_ori[objectname] = {}
		for new_frame, value in new_mapping.items():
			keyframes_ori[objectname][new_frame] = value

	# change position of active keyframe on the timeline
	elif context.window_manager.motion_trail.mode == 'timing' and \
	active_keyframe:
		objectname, frame, frame_ori, action_ob, child = active_keyframe
		mat = inverse_getter(frame, child if child else action_ob, context)

		mouse_ori_world = mat @ screen_to_world(context, drag_mouse_ori[0],
			drag_mouse_ori[1])
		vector = mat @ screen_to_world(context, event.mouse_region_x,
			event.mouse_region_y)
		d = vector - mouse_ori_world

		locs_ori = [[f_ori, coords] for f_mapped, [f_ori, coords] in
					keyframes_ori[objectname].items()]
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
					current = mathutils.Vector(coords)
					next = mathutils.Vector(locs_ori[i + 1][1])
					previous = mathutils.Vector(locs_ori[i - 1][1])
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
		d_screen = mathutils.Vector([event.mouse_region_x,
			event.mouse_region_y]) - drag_mouse_ori
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
		curves = get_curves(action_ob, child)

		for i, curve in enumerate(curves):
			for kf in curve.keyframe_points:
				if abs(kf.co[0] - frame) < 1e-4:
					kf.co[0] = new_frame
					kf.handle_left[0] = handles_ori[objectname][frame_ori]["left"][i][0] + d_frame
					kf.handle_right[0] = handles_ori[objectname][frame_ori]["right"][i][0] + d_frame
					break
		active_keyframe = [objectname, new_frame, frame_ori, action_ob, child]

	# change position of active timebead on the timeline, thus altering speed
	elif context.window_manager.motion_trail.mode == 'speed' and \
	active_timebead:
		objectname, frame, frame_ori, action_ob, child = active_timebead
		mat = inverse_getter(frame, child if child else action_ob, context)

		mouse_ori_world = mat @ screen_to_world(context, drag_mouse_ori[0],
			drag_mouse_ori[1])
		vector = mat @ screen_to_world(context, event.mouse_region_x,
			event.mouse_region_y)
		d = vector - mouse_ori_world

		# determine direction (to next or previous keyframe)
		curves = get_curves(action_ob, child)
		fcx, fcy, fcz = curves
		locx = fcx.evaluate(frame_ori)
		locy = fcy.evaluate(frame_ori)
		locz = fcz.evaluate(frame_ori)
		loc_ori = mathutils.Vector([locx, locy, locz])  # bonespace
		keyframes = [kf for kf in keyframes_ori[objectname]]
		keyframes.append(frame_ori)
		keyframes.sort()
		frame_index = keyframes.index(frame_ori)
		kf_prev = keyframes[frame_index - 1]
		kf_next = keyframes[frame_index + 1]
		vec_prev = (
				(mathutils.Matrix.Translation(-loc_ori) @ mat) @ \
				mathutils.Vector(keyframes_ori[objectname][kf_prev][1])).normalized()
		vec_next = (
				(mathutils.Matrix.Translation(-loc_ori) @ mat) @ \
				mathutils.Vector(keyframes_ori[objectname][kf_next][1])).normalized()
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
											handles_ori[objectname][kf_bead]["left"][i][0] +
											d_frame, kf_bead - 1
											)
						angle = mathutils.Vector([-1, 0]).angle(
											mathutils.Vector(kf.handle_left) -
											mathutils.Vector(kf.co), 0
											)
						if angle != 0:
							angles.append(angle)
					else:
						# right side
						kf.handle_right[0] = max(
											handles_ori[objectname][kf_bead]["right"][i][0] +
											d_frame, kf_bead + 1
											)
						angle = mathutils.Vector([1, 0]).angle(
											mathutils.Vector(kf.handle_right) -
											mathutils.Vector(kf.co), 0
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
		active_timebead = [objectname, bead_frame, frame_ori, action_ob, child]

	return(active_keyframe, active_timebead, keyframes_ori)


# revert changes made by dragging
def cancel_drag(context, active_keyframe, active_handle, active_timebead,
keyframes_ori, handles_ori):
	# revert change in 3d-location of active keyframe and its handles
	if context.window_manager.motion_trail.mode == 'location' and \
	active_keyframe:
		objectname, frame, frame_ori, active_ob, child = active_keyframe
		curves = get_curves(active_ob, child)
		for i, curve in enumerate(curves):
			for kf in curve.keyframe_points:
				if kf.co[0] == frame:
					kf.co[1] = context.window_manager.motion_trail.keyframe_backup[i]
					kf.handle_left[1] = handles_ori[objectname][frame]["left"][i][1]
					kf.handle_right[1] = handles_ori[objectname][frame]["right"][i][1]
					break
		context.window_manager.motion_trail.backed_up_keyframes = False

	# revert change in 3d-location of active handle
	elif context.window_manager.motion_trail.mode == 'location' and \
	active_handle:
		objectname, frame, side, active_ob, child = active_handle
		curves = get_curves(active_ob, child)
		for i, curve in enumerate(curves):
			for kf in curve.keyframe_points:
				if kf.co[0] == frame:
					kf.handle_left[1] = handles_ori[objectname][frame]["left"][i][1]
					kf.handle_right[1] = handles_ori[objectname][frame]["right"][i][1]
					break

	# revert position of all keyframes and handles on timeline
	elif context.window_manager.motion_trail.mode == 'timing' and \
	active_timebead:
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
	elif context.window_manager.motion_trail.mode == 'timing' and \
	active_keyframe:
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
	elif context.window_manager.motion_trail.mode == 'speed' and \
	active_timebead:
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
		objectname, frame, side, action_ob, child = active_keyframe
		side = "both"
	elif active_handle:
		objectname, frame, side, action_ob, child = active_handle
	else:
		# no active handle(s)
		return(False)

	# properties used when changing handle type
	bpy.context.window_manager.motion_trail.handle_type_frame = frame
	bpy.context.window_manager.motion_trail.handle_type_side = side
	bpy.context.window_manager.motion_trail.handle_type_action_ob = \
		action_ob.name
	if child:
		bpy.context.window_manager.motion_trail.handle_type_child = child.name
	else:
		bpy.context.window_manager.motion_trail.handle_type_child = ""

	curves = get_curves(action_ob, child=child)
	for c in curves:
		for kf in c.keyframe_points:
			if kf.co[0] == frame:
				if side in ("left", "both"):
					return(kf.handle_left_type)
				else:
					return(kf.handle_right_type)

	return("AUTO")


# turn the given frame into a keyframe
def insert_keyframe(self, context, frame):
	objectname, frame, frame, action_ob, child = frame
	curves = get_curves(action_ob, child)
	for c in curves:
		y = c.evaluate(frame)
		if c.keyframe_points:
			c.keyframe_points.insert(frame, y)

	bpy.context.window_manager.motion_trail.force_update = True
	calc_callback(self, context)


# change the handle type of the active selection
def set_handle_type(self, context):
	if not context.window_manager.motion_trail.handle_type_enabled:
		return
	if context.window_manager.motion_trail.handle_type_old == \
	context.window_manager.motion_trail.handle_type:
		# function called because of selection change, not change in type
		return
	context.window_manager.motion_trail.handle_type_old = \
		context.window_manager.motion_trail.handle_type

	frame = bpy.context.window_manager.motion_trail.handle_type_frame
	side = bpy.context.window_manager.motion_trail.handle_type_side
	action_ob = bpy.context.window_manager.motion_trail.handle_type_action_ob
	action_ob = bpy.data.objects[action_ob]
	child = bpy.context.window_manager.motion_trail.handle_type_child
	if child:
		child = action_ob.pose.bones[child]
	new_type = context.window_manager.motion_trail.handle_type

	curves = get_curves(action_ob, child=child)
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

	context.window_manager.motion_trail.force_update = True

def update_callback(self, context):
	# Remove handler if file was changed and we lose access to self
	try:
		self.properties
	except:
		if global_mtrail_handler_update:
			bpy.types.SpaceGraphEditor.draw_handler_remove(global_mtrail_handler_update, 'WINDOW')
		return
	
	context.window_manager.motion_trail.force_update = True
	

global_mtrail_handler_calc = None
global_mtrail_handler_draw = None
global_mtrail_handler_update = None

class MotionTrailOperator(bpy.types.Operator):
	bl_idname = "view3d.motion_trail"
	bl_label = "Motion Trail"
	bl_description = "Edit motion trails in 3d-view"
	bl_options = {'REGISTER'}

	_handle_calc = None
	_handle_draw = None
	_handle_update = None

	@staticmethod
	def handle_add(self, context):
		global global_mtrail_handler_calc
		global_mtrail_handler_calc = \
		MotionTrailOperator._handle_calc = bpy.types.SpaceView3D.draw_handler_add(
			calc_callback, (self, context), 'WINDOW', 'POST_VIEW')
		
		global global_mtrail_handler_draw
		global_mtrail_handler_draw = \
		MotionTrailOperator._handle_draw = bpy.types.SpaceView3D.draw_handler_add(
			draw_callback, (self, context), 'WINDOW', 'POST_PIXEL')
		
		global global_mtrail_handler_update
		global_mtrail_handler_update = \
		MotionTrailOperator._handle_update = bpy.types.SpaceGraphEditor.draw_handler_add(
			update_callback, (self, context), 'WINDOW', 'POST_PIXEL')

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
			
		MotionTrailOperator._handle_calc = None
		MotionTrailOperator._handle_draw = None
		MotionTrailOperator._handle_update = None

	def modal(self, context, event):
		# XXX Required, or custom transform.translate will break!
		# XXX If one disables and re-enables motion trail, modal op will still be running,
		# XXX default translate op will unintentionally get called, followed by custom translate.
		if not context.window_manager.motion_trail.enabled:
			MotionTrailOperator.handle_remove()
			if context.area:
				context.area.tag_redraw()
			return {'FINISHED'}

		if not context.area or not context.region or event.type == 'NONE':
			#context.area.tag_redraw()
			return {'PASS_THROUGH'}

		no_passthrough = False

		if (not context.active_object or
				context.active_object.mode not in ('OBJECT', 'POSE')):
			if self.drag:
				self.drag = False
				self.lock = True
				context.window_manager.motion_trail.force_update = True
				context.window_manager.motion_trail.backed_up_keyframes = False
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

		mt = context.window_manager.motion_trail
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
				self.keyframes_ori, self.handles_ori = \
					get_original_animation_data(context, self.keyframes)
				self.drag_mouse_ori = mathutils.Vector([event.mouse_region_x,
					event.mouse_region_y])
				self.drag = True
				self.lock = False
			else:
				# stop drag
				self.drag = False
				self.lock = True
				context.window_manager.motion_trail.force_update = True
				context.window_manager.motion_trail.backed_up_keyframes = False
		elif event.type == self.transform_key and event.value == 'PRESS':
			# call default translate()
			if bpy.ops.transform.translate.poll():
				bpy.ops.transform.translate('INVOKE_DEFAULT')
		elif (event.type == 'ESC' and self.drag and event.value == 'PRESS') or \
			 (event.type == 'RIGHTMOUSE' and self.drag and event.value == 'PRESS'):
			# cancel drag
			self.drag = False
			self.lock = True
			context.window_manager.motion_trail.force_update = True
			context.window_manager.motion_trail.backed_up_keyframes = False
			self.active_keyframe, self.active_timebead = cancel_drag(context,
				self.active_keyframe, self.active_handle,
				self.active_timebead, self.keyframes_ori, self.handles_ori)
		elif event.type == 'MOUSEMOVE' and self.drag:
			# drag
			self.active_keyframe, self.active_timebead, self.keyframes_ori = \
				drag(context, event, self.drag_mouse_ori,
				self.active_keyframe, self.active_handle,
				self.active_timebead, self.keyframes_ori, self.handles_ori)
			no_passthrough = True
		elif event.type in [select, deselect_nohit] and event.value == 'PRESS' and \
		not self.drag and not event.shift and not event.alt and not \
		event.ctrl:
			# select
			threshold = mt.select_threshold
			clicked = mathutils.Vector([event.mouse_region_x,
				event.mouse_region_y])

			context.window_manager.motion_trail.force_update = True
			found = False

			if context.window_manager.motion_trail.path_before == 0:
				frame_min = context.scene.frame_start
			else:
				frame_min = max(
							context.scene.frame_start,
							context.scene.frame_current -
							context.window_manager.motion_trail.path_before
							)
			if context.window_manager.motion_trail.path_after == 0:
				frame_max = context.scene.frame_end
			else:
				frame_max = min(
							context.scene.frame_end,
							context.scene.frame_current +
							context.window_manager.motion_trail.path_after
							)

			for objectname, values in self.click.items():
				if found:
					break
				for frame, type, coord, action_ob, child in values:
					if frame < frame_min or frame > frame_max:
						continue
					if (coord - clicked).length <= threshold:
						found = True

						if event.type == select:
							self.active_keyframe = False
							self.active_handle = False
							self.active_timebead = False
							self.active_frame = False
							context.window_manager.motion_trail.handle_type_enabled = True
							no_passthrough = True

							if type == "keyframe":
								self.active_keyframe = [objectname, frame, frame,
									action_ob, child]
							elif type == "handle_left":
								self.active_handle = [objectname, frame, "left",
									action_ob, child]
							elif type == "handle_right":
								self.active_handle = [objectname, frame, "right",
									action_ob, child]
							elif type == "timebead":
								self.active_timebead = [objectname, frame, frame,
									action_ob, child]
							elif type == "frame":
								self.active_frame = [objectname, frame, frame,
									action_ob, child]
							break
			if not found:
				if event.type == deselect_nohit:
					attrs = ["active_keyframe", "active_handle", "active_timebead", "active_frame"]
					# If a change happens, then no passthrough
					gotten = [getattr(self, attr) for attr in attrs]
					no_passthrough = not reduce(lambda accum, next: accum and not next, gotten, True)
					
					for attr in attrs:
						setattr(self, attr, False)
					context.window_manager.motion_trail.handle_type_enabled = False
					
				pass
			else:
				handle_type = get_handle_type(self.active_keyframe,
					self.active_handle)
				if handle_type:
					context.window_manager.motion_trail.handle_type_old = \
						handle_type
					context.window_manager.motion_trail.handle_type = \
						handle_type
				else:
					context.window_manager.motion_trail.handle_type_enabled = \
						False
		elif event.type == 'LEFTMOUSE' and event.value == 'PRESS' and \
		self.drag:
			# stop drag
			self.drag = False
			self.lock = True
			context.window_manager.motion_trail.force_update = True
			context.window_manager.motion_trail.backed_up_keyframes = False
			bpy.ops.ed.undo_push(message="Confirmed Motion Trail drag")
			no_passthrough = True

		elif event.type == deselect_always and event.value == 'PRESS' and \
		not self.drag and not event.shift and not event.alt and not \
		event.ctrl:
			self.active_keyframe = False
			self.active_handle = False
			self.active_timebead = False
			self.active_frame = False
			context.window_manager.motion_trail.handle_type_enabled = False

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

		if not context.window_manager.motion_trail.enabled:
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
			context.window_manager.motion_trail.force_update = True
			context.window_manager.motion_trail.backed_up_keyframes = False
			context.window_manager.motion_trail.handle_type_enabled = False
			self.cached = {
					"path": {}, "keyframes": {},
					"timebeads_timing": {}, "timebeads_speed": {}
					}

			for kmi in kmis:
				kmi.active = False

			MotionTrailOperator.handle_add(self, context)
			context.window_manager.motion_trail.enabled = True

			if context.area:
				context.area.tag_redraw()

			context.window_manager.modal_handler_add(self)
			bpy.ops.ed.undo_push(message="Started Motion Trail modal operator")
			return {'RUNNING_MODAL'}

		else:
			# disable
			for kmi in kmis:
				kmi.active = True
			MotionTrailOperator.handle_remove()
			context.window_manager.motion_trail.enabled = False

			if context.area:
				context.area.tag_redraw()

			return {'FINISHED'}

def load_defaults(context):
	prefs = context.preferences.addons[__name__].preferences
	for p in configurable_props:
		default = getattr(prefs.default_trail_settings, p)
		setattr(context.window_manager.motion_trail, p, default)

class MotionTrailLoadDefaults(bpy.types.Operator):
	bl_idname="view3d.motion_trail_load_defaults"
	bl_label="Load Defaults"
	bl_description="Reset all the current settings to match what's in the addon's preferences"
	
	def execute(self, context):
		load_defaults(context)
		return {'FINISHED'}

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
		if (not context.window_manager.motion_trail.loaded_defaults):
			load_defaults(context)
			context.window_manager.motion_trail.loaded_defaults = True
		col = self.layout.column()
		if not context.window_manager.motion_trail.enabled:
			col.operator("view3d.motion_trail", text="Enable motion trail")
		else:
			col.operator("view3d.motion_trail", text="Disable motion trail")

		box = self.layout.box()
		box.prop(context.window_manager.motion_trail, "mode")
		# box.prop(context.window_manager.motion_trail, "calculate")
		if context.window_manager.motion_trail.mode == 'timing':
			box.prop(context.window_manager.motion_trail, "timebeads")

		box = self.layout.box()
		col = box.column()
		row = col.row()

		if context.window_manager.motion_trail.path_display:
			row.prop(context.window_manager.motion_trail, "path_display",
				icon="DOWNARROW_HLT", text="", emboss=False)
		else:
			row.prop(context.window_manager.motion_trail, "path_display",
				icon="RIGHTARROW", text="", emboss=False)

		row.label(text="Path options")

		if context.window_manager.motion_trail.path_display:
			col.prop(context.window_manager.motion_trail, "path_style",
				text="Style")
			
			if context.window_manager.motion_trail.path_style == 'simple':
				col.row().prop(context.window_manager.motion_trail, "simple_color")
			elif context.window_manager.motion_trail.path_style == 'speed':
				col.row().prop(context.window_manager.motion_trail, "speed_color_min")
				col.row().prop(context.window_manager.motion_trail, "speed_color_max")
			else:
				col.row().prop(context.window_manager.motion_trail, "accel_color_neg")	
				col.row().prop(context.window_manager.motion_trail, "accel_color_static")
				col.row().prop(context.window_manager.motion_trail, "accel_color_pos")
				
			grouped = col.column(align=True)
			grouped.prop(context.window_manager.motion_trail, "path_width",
				text="Width")
			grouped.prop(context.window_manager.motion_trail,
				"path_resolution")
			row = grouped.row(align=True)
			row.prop(context.window_manager.motion_trail, "path_before")
			row.prop(context.window_manager.motion_trail, "path_after")
			col = col.column(align=True)
			col.prop(context.window_manager.motion_trail, "keyframe_numbers")
			if context.window_manager.motion_trail.keyframe_numbers:
				col.row().prop(context.window_manager.motion_trail, "text_color")
				col.row().prop(context.window_manager.motion_trail, "selected_text_color")
			col.prop(context.window_manager.motion_trail, "frame_display")
			if context.window_manager.motion_trail.frame_display:
				col.row().prop(context.window_manager.motion_trail, "frame_color")

		box = self.layout.box()
		col = box.column(align=True)
		if context.window_manager.motion_trail.mode == 'location':
			col.prop(context.window_manager.motion_trail, "handle_display",
				text="Handles")
			if context.window_manager.motion_trail.handle_display:
				row = col.row()
				row.enabled = context.window_manager.motion_trail.\
					handle_type_enabled
				row.prop(context.window_manager.motion_trail, "handle_type")
				col.prop(context.window_manager.motion_trail, "handle_direction")
				col.prop(context.window_manager.motion_trail, "handle_length")
				
				col.row().prop(context.window_manager.motion_trail, "handle_color")
				col.row().prop(context.window_manager.motion_trail, "handle_line_color")
				col.row().prop(context.window_manager.motion_trail, "selection_color_dark")
		else:
			col.row().prop(context.window_manager.motion_trail, "timebead_color")

		box = self.layout.box()
		col = box.column(align=True)
		col.row().prop(context.window_manager.motion_trail, "selection_color")
		col.row().prop(context.window_manager.motion_trail, "select_key")
		col.row().prop(context.window_manager.motion_trail, "select_threshold")
		col.row().prop(context.window_manager.motion_trail, "deselect_nohit_key")
		col.row().prop(context.window_manager.motion_trail, "deselect_always_key")
		col.label(text="For the time being, confirm/cancel")
		col.label(text="is LMB/RMB or Esc")
			
		self.layout.column().operator("view3d.motion_trail_load_defaults")

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
		
	keyframe_backup: FloatVectorProperty()
	backed_up_keyframes: BoolProperty(default=False)

	handle_type_enabled: BoolProperty(default=False)
	handle_type_frame: FloatProperty()
	handle_type_side: StringProperty()
	handle_type_action_ob: StringProperty()
	handle_type_child: StringProperty()

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
			("location", "Location", "Change path that is followed"),
			("speed", "Speed", "Change speed between keyframes"),
			("timing", "Timing", "Change position of keyframes on timeline")),
			description="Enable editing of certain properties in the 3d-view",
			default='location',
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
	path_resolution: IntProperty(name="Resolution",
			description="10 is smoothest, but could be "
						"slow when adjusting keyframes, handles or timebeads",
			default=10,
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
			items=(
			("time", "Time", "Use only the time coordinate of the handles"),
			("wtime", "Weighted Time", "0.75*time + 0.25*location"),
			("location", "Location", "Use only the location coordinate of the handles"),
			("wloc", "Weighted Location", "0.25*time + 0.75*location"),
			("len", "Directional length", "Use the length of the handle, positive for right and negative for left")),
			default='wtime'
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
	handle_color: FloatVectorProperty(name="Handle color",
			description="Color that unselected handles will be colored in",
			default=(1.0, 1.0, 0.0, 1.0),
			min=0.0, soft_max=1.0,
			size=4,
			subtype='COLOR'
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
			
configurable_props = ["select_key", "select_threshold", "deselect_nohit_key", "deselect_always_key", "mode", "path_style", 
"simple_color", "speed_color_min", "speed_color_max", "accel_color_neg", "accel_color_static", "accel_color_pos",
"keyframe_color", "frame_color", "selection_color", "selection_color_dark", "handle_color", "handle_line_color", "timebead_color", 
"text_color", "selected_text_color", "path_width", "path_resolution", "path_before", "path_after",
"keyframe_numbers", "frame_display", "handle_display", "handle_length", "handle_direction"]
			
class MotionTrailPreferences(bpy.types.AddonPreferences):
	bl_idname = __name__
	
	default_trail_settings: PointerProperty(type=MotionTrailProps)
	
	def draw(self, context):
		layout = self.layout
		col = layout.column()
		col.label(text=DESELECT_WARNING)
		col.label(text="Default values for all settings:")
		col.label(text="")
		for p in configurable_props:
			col.row().prop(self.default_trail_settings, p)

classes = (
		MotionTrailProps,
		MotionTrailOperator,
		MotionTrailPanel,
		MotionTrailPreferences,
		MotionTrailLoadDefaults,
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
