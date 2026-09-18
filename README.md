# BillardBallRayTracing3D
A couple of caveats to all of this

1. To intersect the surface is divided into triangles, and then those triangles are used to
cheapy find the intial guess for newtons method. This can fail at high curvature points. A more 
robust, but significantly more expensive way would be to do newtons method on a grid the (u,v,t) grid.
2. If newtons method fails to converge or if a ray goes inside the obstacle that an error is thrown. Usually
this is caught and the method continues forward. No attempt is made to fix it.