# Quaternion

## Quick definition

A quaternion is written as

$$
q = (w, x, y, z) = w + xi + yj + zk
$$

- `w`: scalar part
- `(x, y, z)`: vector part

Basic multiplication rules:

$$
i^2 = j^2 = k^2 = ijk = -1
$$

$$
ij = k, \quad jk = i, \quad ki = j
$$

$$
ji = -k, \quad kj = -i, \quad ik = -j
$$

So quaternion multiplication is associative but not commutative.

## Quaternion inverse

The inverse of `q` is the quaternion that exactly undoes the rotation/action of `q`.

$$
q q^{-1} = q^{-1} q = 1
$$

where the identity quaternion is

$$
1 = (1, 0, 0, 0)
$$

For

$$
q = (w, x, y, z)
$$

the inverse is

$$
q^{-1} = \frac{(w, -x, -y, -z)}{w^2 + x^2 + y^2 + z^2}
$$

## Conjugate and norm

The conjugate is

$$
q^* = (w, -x, -y, -z)
$$

and

$$
|q|^2 = w^2 + x^2 + y^2 + z^2
$$

So

$$
q^{-1} = \frac{q^*}{|q|^2}
$$

($q^{-1}$ exists when $q \neq 0$.)

## Unit quaternion (for 3D rotation)

In 3D graphics/robotics, rotation quaternions are usually unit quaternions:

$$
|q| = 1
$$

Then

$$
q^{-1} = q^*
$$

So inverse is just sign flip of the vector part.

Example:

$$
q = (0.707, 0, 0, 0.707), \qquad q^{-1} = (0.707, 0, 0, -0.707)
$$

This means same rotation magnitude, opposite direction.

## Rotation formula for a vector

Represent a vector `v = (vx, vy, vz)` as a pure quaternion:

$$
v_{\text{quat}} = (0, v_x, v_y, v_z)
$$

Then rotate by

$$
v' = q\, v_{\text{quat}}\, q^{-1}
$$

where `q` is a unit quaternion. The output `v'` is the rotated vector (read its vector part).

## Geometric meaning

If `q` rotates by $+\theta$ around an axis, $q^{-1}$ rotates by $-\theta$ around the same axis.

Axis-angle form:

$$
q = \left(\cos\frac{\theta}{2},\, \sin\frac{\theta}{2}\, \mathbf{u}\right)
$$

with unit axis `u`. Then

$$
q^{-1} = \left(\cos\frac{\theta}{2},\, -\sin\frac{\theta}{2}\, \mathbf{u}\right)
$$

which is exactly the opposite rotation.

## Quaternion summary

$$
q = (w, x, y, z), \qquad
q^{-1} = \frac{(w, -x, -y, -z)}{w^2 + x^2 + y^2 + z^2}
$$

For unit quaternion:

$$
q^{-1} = q^*
$$

Interpretation: the inverse quaternion is the exact undo rotation.

## Dual quaternion (rotation + translation)

A dual quaternion represents rigid 3D motion (rotation + translation) in one form:

$$
Q = q_r + \varepsilon q_d
$$

- $q_r$: rotation quaternion
- $q_d$: translation-related part
- $\varepsilon$: dual unit with $\varepsilon^2 = 0$

Why it is useful: a standard quaternion encodes rotation only, but most 3D transforms need both rotation and translation at the same time.

For a point/vector transform, the result has the form:

$$
v' = R(v) + t
$$

So one dual-quaternion transform gives a rigid transform directly, without introducing scale or shear.
