# Conventions

## Euler Angles and Rotations

In `qdef` Euler angles are usually labeled with the three greek letters $\alpha$, $\beta$, and $\gamma$. These angles represent the following sequence of rotations about fixed z-y-z axes:

- first a rotation about the z-axis by an angle $\alpha$, 
- then a rotation about the y-axis by an angle $\beta$,
- and finally a rotation about the z-axis by an angle $\gamma$.

$R(\alpha, \beta, \gamma) = R_z(\gamma) * R_y(\beta) * R_z(\alpha)$

Furthermore, it is assumed that the rotations are *active* operations, in the sense that the relevant physical system is assumed to be fixed, so that the rotation is of the coordinate system used for its description.
