---
name: spot-fetch
description: Orchestrates the Spot robot to fetch objects from named locations. Use when asked to "go get X", "bring me X from Y", "fetch X", or any task involving navigating to a place and picking up an object.
always: false
---

# Spot Fetch Skill

Use this skill whenever the user asks the Spot robot to go somewhere and retrieve an object.

## When to use

- "Vá para a mesa e me traga o martelo"
- "Busque a garrafa na prateleira"
- "Traga o item X do lugar Y"
- "Fetch the [object] from [location]"

## Tools available

- `spot_location` — navigate to named places, save/list locations
- `spot_base` — move the base (forward/backward, rotate, navigate)
- `spot_arm` — arm control and grasp pipeline (segment → grasps → plan → execute)
- `spot_perception` — camera status check

## Full fetch sequence

Execute these steps **in order**. Do not skip steps. If any step fails, report the error clearly and stop.

### Step 0 — Check cameras
Before anything else, verify cameras are ready:
```
spot_perception(action="camera_status")
```
If cameras are NOT ready, stop and inform the user.

### Step 1 — Save current position as "origem"
Before navigating away, save where you are now so you can return:
```
spot_location(action="save_current_as", name="origem", description="posição antes do fetch")
```

### Step 2 — Navigate to the location
Go to the named location where the object is:
```
spot_location(action="go_to_location", name="<location_name>")
```
If the location is not found, list known locations and ask the user to clarify or to move the robot there and use `save_current_as`.

### Step 3 — Move arm to observe pose
Position the arm so the hand camera has a clear view of the workspace:
```
spot_arm(action="move_to_observe")
```

### Step 4 — Run the grasp pipeline
Segment the object, generate grasps, plan trajectory, open gripper, execute:
```
spot_arm(action="open_gripper")
spot_arm(action="full_grasp_pipeline", object_name="<object_name>")
```
The pipeline handles: segmentation → Contact GraspNet → cuRobo planning → arm movement → gripper close.

If the pipeline fails at segmentation: the object may not be visible. Try rotating the base slightly and retry.
If the pipeline fails at grasps: try moving the arm to a slightly different observe pose.
If the pipeline fails at cuRobo: try generate_grasps again (different pose candidate).

### Step 5 — Move arm to carry pose
After successful grasp, tuck the arm so the object is secure during navigation:
```
spot_arm(action="move_to_carry")
```

### Step 6 — Navigate back to origin
Return to where the user is:
```
spot_location(action="go_to_location", name="origem")
```

### Step 7 — Present the object
Move arm to home/present position and open gripper to hand off:
```
spot_arm(action="move_to_home")
```
Inform the user that the object has been retrieved.

## Error handling

| Failure | Action |
|---|---|
| Camera not ready | Check `/camera/hand/image_raw` topic. Ask user to check Spot driver. |
| Location not found | List known locations. Ask user to navigate manually and `save_current_as`. |
| Segmentation failed | Object may be occluded or name incorrect. Retry after rotating base. |
| Grasp generation failed | Try `move_to_observe` again from slightly different angle. |
| cuRobo planning failed | Retry `generate_grasps` then `plan_trajectory`. |
| Arm motion failed | Run `move_to_home` for safety, then report failure. |
| Navigation failed | Check Nav2 stack and map. Try smaller intermediate waypoints. |

## Important notes

- Always save "origem" **before** navigating (Step 1)
- Always `move_to_carry` **before** navigating with object (Step 5)  
- If the task is aborted for any reason, always call `move_to_home` to put the arm in a safe position
- The grasp pipeline is stateful: segment_object → generate_grasps → plan_trajectory → execute_grasp must run in order when called individually
- `full_grasp_pipeline` handles all 4 phases automatically and is preferred

## Example — "vá para a mesa e me traga o martelo"

1. `spot_perception(action="camera_status")` → cameras ready ✓
2. `spot_location(action="save_current_as", name="origem")`
3. `spot_location(action="go_to_location", name="mesa")`
4. `spot_arm(action="move_to_observe")`
5. `spot_arm(action="open_gripper")`
6. `spot_arm(action="full_grasp_pipeline", object_name="martelo")`
7. `spot_arm(action="move_to_carry")`
8. `spot_location(action="go_to_location", name="origem")`
9. `spot_arm(action="move_to_home")`
→ "Trouxe o martelo da mesa!"