# This addon is currently potentially very buggy, and is still in development.
 Expect changes frequently, or rarely.
 
 This is an update of the old motion trail addon by Bart Crouch.
 
 The features present in it before (basically, only motion trails for objects) should work.<br>
 Selection may be a bit weird.<br>
 More customizable colors and a settings menu in the addons list have been added.<br>
 I'm attempting to add support for bones and parented objects (and it seems so did Crouch).<br>
 Trails and editing the trails for parented objects should work! Bones should work well if there is no rotation, and for some reason unknown to me will be slightly offset if there is rotation.<br>
 There is initial support for child of constraints. Interpolation and disabling location/rotation/scale doesn't seem to work properly. No idea why.<br>
 Rotation and scale's effects on motion trails are slightly broken for everything.<br>
 **The addon tends to kill undos.** I'd say "save often", but that is something you should be doing anyways.
 
 To get the addon, go to the `src` folder, click on the only python file there, click on raw, and save that.
 After installed, the addon is under the "Testing" tab.
 
 https://raw.githubusercontent.com/a-One-Fan/Blender-Motion-Trail-Update/one/src/animation_motion_trail_updated.py