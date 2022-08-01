# This addon is currently potentially very buggy, and is still in development.
 Expect changes frequently, or rarely.
 
 This is an update of the old motion trail addon by Bart Crouch.
 
 The features present in it before (basically, only motion trails for objects) work.<br>
 Selection may be a bit weird.<br>
 I'm attempting to add support for bones and parented objects (and it seems so did Crouch), support is there for trails for parented objects and for unparented bones.<br>
 Trails of parented bones are offset. I haven't figured out what matrices I need yet.<br>
 Moving the keyframes of parented objects or any bones around will cause them to teleport initially as well as slightly break handles in the f-curve editor. Also due to unfigured matrices.<br>
 **The addon tends to kill undos.** I'd say "save often", but that is something you should be doing anyways.
 
 To get the addon, go to the src folder, click on the only python file there, click on raw, and save that.
 After installed, the addon is under the "Testing" tab.
 
 https://raw.githubusercontent.com/a-One-Fan/Blender-Motion-Trail-Update/one/src/animation_motion_trail_updated.py