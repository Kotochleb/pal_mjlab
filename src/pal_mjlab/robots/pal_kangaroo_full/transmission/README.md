# Transmission lookup tables

The tables use right-leg coordinates. Both robot models use the joint signs
defined in `pal_kangaroo_full/kangaroo_full_constants.py` to query these same
tables for the left leg.

## Hip XY actuator order

`hip_xy_jacobian_map.npz` follows the actuator names in
`pal_kangaroo_full/xmls/kangaroo_full_tendons.xml`:

| LUT row | Actuator | Right-leg tendon endpoints |
| --- | --- | --- |
| 0 | `leg_right_2_actuator` | `right_hip_xy_r_slider_connect_a/b` |
| 1 | `leg_right_3_actuator` | `right_hip_xy_l_slider_connect_a/b` |

The `J` rows, `actuators` position components and `forces` sample components
have all been swapped from the original generated table. The joint columns,
grid axes and `actuator_names` remain in numeric order. The
`cache_actuator_permutation` field records the `[1, 0]` permutation from the
original cache, and `actuator_order_source` records the reference model.

Apply this actuator permutation when regenerating from the original cache.
Do not swap the joint columns or also swap outputs in the controller.
The geometry tests compare the interpolated table with the actual tendon
length derivatives and the full linkage's closed-chain derivatives.
