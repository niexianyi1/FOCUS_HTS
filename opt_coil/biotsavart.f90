







SUBROUTINE biot_savart(pos, coilxyz, current, dl, bfield, nseg)
   ! Calculate magnetic field using the Biot-Savart Law
   ! (the close point doesn't have to be repeated!)
   !
   ! input params:
   !       pos(3): double, positions to be evaluated
   !       coilxyz(nseg,3): double, xyz points for the coil
   !       current: double, coil current
   !       dl(nseg,3): double, tangent vector (dx, dy, dz)
   !       nseg: int, optional, number of coil segments
   ! output params:
   !       bfield(3): double, B-vec at the evaluation points
   IMPLICIT NONE

   INTEGER, INTENT(IN) :: npos, nseg
   REAL*8, INTENT(IN) :: pos(3), coilxyz(nseg, 3), current, dl(nseg, 3)
   REAL*8, INTENT(OUT) :: bfield(3)

   INTEGER :: i, j
   REAL*8 :: x, y, z, lx, ly, lz, rm3, Bx, By, Bz
   REAL*8, PARAMETER :: mu0_over_4pi = 1.0E-7

    x = pos(1); y = pos(2); z = pos(3)
    Bx = 0; By = 0; Bz = 0
    DO j = 1, nseg
        lx = x - coilxyz(j, 1)
        ly = y - coilxyz(j, 2)
        lz = z - coilxyz(j, 3)
        rm3 = (sqrt(lx**2 + ly**2 + lz**2))**(-3)
        Bx = Bx + (lz*dl(j, 2) - ly*dl(j, 3))*rm3
        By = By + (lx*dl(j, 3) - lz*dl(j, 1))*rm3
        Bz = Bz + (ly*dl(j, 1) - lx*dl(j, 2))*rm3
    END DO
    bfield(i, 1) = Bx
    bfield(i, 2) = By
    bfield(i, 3) = Bz


   bfield = bfield*mu0_over_4pi*current

   RETURN









