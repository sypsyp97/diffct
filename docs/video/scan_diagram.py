"""Orthographic CT diagram with per-pixel depth tests against the measured surface.

All points share make_walnut_3d's camera and physical coordinate system. Depth
increases toward the viewer. No front/back decision is made from object centres.
"""
import numpy as np
from PIL import Image

BG = np.array([247.,247.,242.])
GREEN = np.array([40.,118.,92.])
GREY = np.array([95.,107.,100.])
LIGHT = np.array([201.,208.,201.])


class ScanDiagram:
    def __init__(self, surface, depth, width=960):
        self.width, self.height = width, int(width*.8)
        self.extent = np.array([2.8,2.24])
        az, el = np.deg2rad([-35,22])
        ex=np.array([np.cos(az),-np.sin(az),0])
        ey=np.array([np.sin(el)*np.sin(az),np.sin(el)*np.cos(az),np.cos(el)])
        self.basis=np.stack([ex,ey,np.cross(ex,ey)])
        self.base=np.full((self.height,self.width,3),BG,dtype=np.float32)
        self.surface_depth=np.full((self.height,self.width),-np.inf,dtype=np.float32)
        # Match the surface's orthographic extent to this larger camera viewport.
        side=round(2.7/(2*self.extent[0])*self.width)
        rgba=np.asarray(Image.fromarray(surface).resize((side,side),Image.Resampling.BILINEAR))
        indices=np.linspace(0,len(depth)-1,side).round().astype(int)
        z=depth[np.ix_(indices,indices)]
        y0=(self.height-side)//2;x0=(self.width-side)//2
        a=rgba[...,3:4]/255
        self.base[y0:y0+side,x0:x0+side]=rgba[...,:3]*a+BG*(1-a)
        self.surface_depth[y0:y0+side,x0:x0+side]=np.where(rgba[...,3]>127,z,-np.inf)
        self.clear()

    def clear(self):
        self.rgb=self.base.copy()
        self.z=self.surface_depth.copy()

    def project(self, points):
        q=np.asarray(points)@self.basis.T
        return np.stack([(q[...,0]/self.extent[0]+1)*(self.width-1)/2,
                         (1-q[...,1]/self.extent[1])*(self.height-1)/2,q[...,2]],axis=-1)

    def patch(self, points, margin=2):
        lo=np.floor(np.min(points[:,:2],axis=0)-margin).astype(int)
        hi=np.ceil(np.max(points[:,:2],axis=0)+margin).astype(int)
        lo=np.maximum(lo,0);hi=np.minimum(hi,[self.width-1,self.height-1])
        if np.any(hi<lo):return None
        yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        return (slice(lo[1],hi[1]+1),slice(lo[0],hi[0]+1)),xx,yy

    def paint(self, sl, depth, coverage, color, opacity=1):
        visible=(depth>=self.z[sl]-1e-4)&(coverage>0)
        alpha=np.where(visible,coverage*opacity,0)[...,None]
        self.rgb[sl]=self.rgb[sl]*(1-alpha)+color*alpha
        self.z[sl]=np.where(visible&(coverage>.5),np.maximum(depth,self.z[sl]),self.z[sl])

    def line(self, a, b, color=GREEN, width=2, opacity=1):
        a,b=self.project([a,b]);patch=self.patch(np.stack([a,b]),width+1)
        if patch is None:return
        sl,x,y=patch;delta=b[:2]-a[:2];den=delta@delta
        t=np.clip(((x-a[0])*delta[0]+(y-a[1])*delta[1])/max(den,1e-12),0,1)
        dist=np.hypot(x-a[0]-t*delta[0],y-a[1]-t*delta[1])
        depth=a[2]+t*(b[2]-a[2])
        self.paint(sl,depth,np.clip(width/2+.5-dist,0,1),color,opacity)

    def triangle(self, points, opacity=.25):
        pts=self.project(points);patch=self.patch(pts)
        if patch is None:return
        sl,x,y=patch;a,b,c=pts
        den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den)<1e-9:return
        u=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/den
        v=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/den
        w=1-u-v
        inside=(u>=0)&(v>=0)&(w>=0)
        self.paint(sl,u*a[2]+v*b[2]+w*c[2],inside.astype(float),LIGHT,opacity)

    def dot(self, point, radius=.045):
        p=self.project(point);r=radius*self.width/(2*self.extent[0]);patch=self.patch(p[None],r+1)
        if patch is None:return
        sl,x,y=patch;distance=np.hypot(x-p[0],y-p[1])
        depth=p[2]+np.sqrt(np.maximum(radius**2-(distance*2*self.extent[0]/self.width)**2,0))
        self.paint(sl,depth,np.clip(r+.5-distance,0,1),GREEN)

    def draw(self, points, source, discrete=False, strength=1):
        self.clear()
        corners=[np.array([x,y,z]) for x in (-1,1) for y in (-1,1) for z in (-1,1)]
        for i,a in enumerate(corners):
            for b in corners[i+1:]:
                if np.count_nonzero(a-b)==1:self.line(a,b,LIGHT,width=1,opacity=.8)
        radial=source/np.linalg.norm(source)
        u=np.cross([0,0,1],radial);u/=np.linalg.norm(u)
        v=np.cross(radial,u);centre=-1.8*radial
        cs=np.array([centre+.65*su*u+.75*sv*v for su,sv in [(-1,-1),(1,-1),(1,1),(-1,1)]])
        # Detector face, lines and walnut participate in the same depth test.
        self.triangle(cs[[0,1,2]],opacity=.28*strength)
        self.triangle(cs[[0,2,3]],opacity=.28*strength)
        for q in cs:self.line(source,q,width=1.25,opacity=.44*strength)
        edge_bias=self.basis[2]*.015  # one raster-pixel depth bias avoids coplanar edge z-fighting
        for a,b in zip(cs,np.roll(cs,-1,axis=0)):self.line(a+edge_bias,b+edge_bias,GREY,width=1.8,opacity=strength)
        for axis in (u,v):
            tip=centre+.5*axis
            bias=self.basis[2]*.002
            self.line(centre+bias,tip+bias,GREY,width=2,opacity=strength)
            for side in (-1,1):
                wing=tip-.10*axis+side*.04*np.cross(radial,axis)
                self.line(tip+bias,wing+bias,GREY,width=2,opacity=strength)
        if discrete:
            for p in points:self.dot(p)
        else:
            for a,b in zip(points,points[1:]):self.line(a,b,width=2.7,opacity=.95*strength)
        self.dot(source,.067)
        return np.clip(self.rgb,0,255).astype(np.uint8)
