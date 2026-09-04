"""Analytic, knot-sided geometry and equal-resource curve families for Paper A v2.

The original seven-variable quadratic NURBS definition is retained; only its
redundant common spacing scale is removed in the six-dimensional search map.
No archived evaluator is modified.
"""
from __future__ import annotations
from dataclasses import dataclass
import sys
from pathlib import Path
import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import BSpline
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, brentq
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'project'/'vendor'))
sys.path.insert(0,str(ROOT/'project'/'src'))
from cbopt.evaluator import _control_points, _clamped_uniform_knots

@dataclass
class Curve:
    family: str
    parameters: np.ndarray
    pitch: float=7.9
    height: float=3.0

    def __post_init__(self):
        self.parameters=np.asarray(self.parameters,dtype=float)
        if self.family=='nurbs':
            cp=_control_points(self.parameters)
            r=0.0103*np.exp(9.17*self.parameters[5:7])+0.1
            w=np.array([1,r[0],r[0],r[1],r[1],r[0],r[0],r[1],r[1],1])
            k=_clamped_uniform_knots(10,2)
            self.num=BSpline(k,cp*w[:,None],2)
            self.den=BSpline(k,w,2)
            ref=self.num(np.array([.25,.5,.75]))/self.den(np.array([.25,.5,.75]))[:,None]
            self.scale=np.array([self.pitch/(ref[2,0]-ref[0,0]),self.height/(ref[0,1]-ref[1,1])])
            self.offset=np.array([ref[0,0],ref[1,1]])
            self.breaks=np.array([.25,.375,.5,.625,.75])
        elif self.family in ('sine','fourier'):
            self.breaks=np.array([0.,.5,1.])
        else: raise ValueError(self.family)

    def evaluate(self,u,derivative=0):
        u=np.asarray(u,dtype=float)
        if self.family=='nurbs':
            n=self.num(u);d=self.den(u)
            if derivative==0:return (n/d[...,None]-self.offset)*self.scale
            dn=self.num.derivative(1)(u);dd=self.den.derivative(1)(u)
            v=(dn/d[...,None]-n*dd[...,None]/d[...,None]**2)
            if derivative==1:return v*self.scale
            d2n=self.num.derivative(2)(u);d2d=self.den.derivative(2)(u)
            return (d2n/d[...,None]-2*dn*dd[...,None]/d[...,None]**2-n*d2d[...,None]/d[...,None]**2+2*n*dd[...,None]**2/d[...,None]**3)*self.scale
        a,b=(0.,0.) if self.family=='sine' else self.parameters
        theta=2*np.pi*u
        if derivative==0:
            y=self.height/2*(1+np.cos(theta)+a*(np.cos(2*theta)-1)+b*(np.cos(3*theta)-np.cos(theta)))
            return np.stack((self.pitch*u,y),axis=-1)
        if derivative==1:
            dy=-self.height*np.pi*((1-b)*np.sin(theta)+2*a*np.sin(2*theta)+3*b*np.sin(3*theta))
            return np.stack((np.full_like(u,self.pitch),dy),axis=-1)
        ddy=-2*self.height*np.pi**2*((1-b)*np.cos(theta)+4*a*np.cos(2*theta)+9*b*np.cos(3*theta))
        return np.stack((np.zeros_like(u),ddy),axis=-1)

    def arc_length(self):
        return sum(quad(lambda u:float(np.linalg.norm(self.evaluate(u,1))),a,b,epsabs=1e-10,epsrel=1e-11)[0] for a,b in zip(self.breaks[:-1],self.breaks[1:]))

    def curvature(self,u):
        v=self.evaluate(u,1);a=self.evaluate(u,2)
        return np.abs(v[...,0]*a[...,1]-v[...,1]*a[...,0])/np.sum(v*v,axis=-1)**1.5

    def radius(self):
        """Floating-point analytic extrema, including each side of every knot.

        On a rational-quadratic span C=N/w, set A=N'w-Nw',
        K=w^2 det(A,A'), D=A.A. The signed curvature is K/D^(3/2);
        its stationary points solve 2 K' D - 3 K D'=0.
        """
        candidates=[]
        if self.family=='nurbs':
            for left,right in zip(self.breaks[:-1],self.breaks[1:]):
                z=np.array([0.,.5,1.]);u=left+(right-left)*z
                n=self.num(u)*self.scale;w=self.den(u)
                px=Polynomial(np.polynomial.polynomial.polyfit(z,n[:,0],2))
                py=Polynomial(np.polynomial.polynomial.polyfit(z,n[:,1],2))
                pw=Polynomial(np.polynomial.polynomial.polyfit(z,w,2))
                ax=px.deriv()*pw-px*pw.deriv();ay=py.deriv()*pw-py*pw.deriv()
                K=pw*pw*(ax*ay.deriv()-ay*ax.deriv());D=ax*ax+ay*ay
                stationary=(2*K.deriv()*D-3*K*D.deriv()).trim(tol=1e-11)
                roots=stationary.roots() if stationary.degree()>0 else []
                points=[0.,1.]+[float(r.real) for r in roots if abs(r.imag)<1e-7 and -1e-9<=r.real<=1+1e-9]
                for q in points:
                    q=float(np.clip(q,0,1));k=abs(K(q))/D(q)**1.5
                    candidates.append((float(k),float(left+q*(right-left))))
        else:
            grid=np.linspace(0,1,257);vals=self.curvature(grid)
            candidates.extend(zip(vals,grid))
            for i in range(1,len(grid)-1):
                if vals[i]>=vals[i-1] and vals[i]>=vals[i+1]:
                    r=minimize_scalar(lambda u:-float(self.curvature(u)),bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':1e-14})
                    candidates.append((-r.fun,r.x))
        k,u=max(candidates)
        return float(1/k),float(u)

    def dense_radius(self,n=4097):
        grids=[np.linspace(np.nextafter(a,b),np.nextafter(b,a),n) for a,b in zip(self.breaks[:-1],self.breaks[1:])]
        k=np.max(self.curvature(np.concatenate(grids)))
        return float(1/k)

    def monotone_halves(self):
        if self.family=='nurbs':return True
        return bool(np.all(self.evaluate(np.linspace(0,.5,1001),1)[:,1]<=1e-10))

    def nodes(self,elements):
        # Preserve exact crown/trough nodes for every mesh and family.
        lengths=np.array([quad(lambda u:float(np.linalg.norm(self.evaluate(u,1))),a,b,epsabs=1e-10)[0] for a,b in zip([self.breaks[0],.5],[.5,self.breaks[-1]])])
        ideal=(elements-2)*lengths/lengths.sum();counts=np.ones(2,dtype=int)+np.floor(ideal).astype(int)
        for j in np.argsort(-(ideal-np.floor(ideal)))[:elements-counts.sum()]:counts[j]+=1
        out=[]
        for i,(a,b) in enumerate(zip([self.breaks[0],.5],[.5,self.breaks[-1]])):
            u=np.linspace(a,b,max(4001,100*elements));xy=self.evaluate(u)
            s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
            st=np.linspace(0,s[-1],counts[i]+1)
            sub=np.column_stack([np.interp(st,s,xy[:,j]) for j in range(2)])
            out.append(sub if i==0 else sub[1:])
        nodes=np.vstack(out);nodes[0]=[0,self.height];nodes[counts[0],1]=0.;nodes[-1]=[self.pitch,self.height]
        return nodes,self.arc_length()


def from_unit(z,family='nurbs',pitch=7.9,height=3.):
    z=np.asarray(z,dtype=float)
    if family=='nurbs':
        # Four log-spacing ratios to the fifth remove common-scale redundancy.
        p=np.r_[np.exp(4*(z[:4]-.5)),1.,z[4:6]]
    elif family=='fourier':p=np.array([.24*(z[0]-.5),.16*(z[1]-.5)])
    else:p=np.zeros(0)
    return Curve(family,p,pitch,height)


def metrics(curve,thickness):
    L=curve.arc_length();R,u=curve.radius();P,H=curve.pitch,curve.height
    return {'pitch_mm':P,'height_mm':H,'thickness_mm':thickness,'arc_length_mm':L,'radius_min_mm':R,'radius_location_u':u,
        'medium_volume_per_area_mm':thickness*L/P,'equal_t_liner_volume_per_area_mm':thickness*(L/P+2),
        'board_material_fraction':thickness*(L/P+2)/H,'core_material_fraction':thickness*L/(P*H),
        'radius_screen_pass':bool(R>=.9),'monotone_halves':bool(curve.monotone_halves()),'geometry_feasible':bool(R>=.9 and curve.monotone_halves()),'curvature_index_t_over_Rmin':thickness/R}

class PerturbedCurve(Curve):
    """Small smooth forming perturbation preserving the reference extrema.

    y_eta(u)=y(u)+eta H sin^2(2 pi q), q=(u-u0)/(u1-u0).
    Analytic derivatives are retained; stationary curvature candidates are
    bracketed numerically within every original smooth span.
    """
    def __init__(self,base,eta):
        self.base=base;self.eta=float(eta);self.pitch=base.pitch;self.height=base.height
        self.family=base.family;self.parameters=base.parameters;self.breaks=base.breaks
    def evaluate(self,u,derivative=0):
        u=np.asarray(u,dtype=float);out=self.base.evaluate(u,derivative).copy()
        span=self.breaks[-1]-self.breaks[0];q=(u-self.breaks[0])/span
        if derivative==0:f=.5*(1-np.cos(4*np.pi*q))
        elif derivative==1:f=2*np.pi/span*np.sin(4*np.pi*q)
        else:f=8*np.pi**2/span**2*np.cos(4*np.pi*q)
        out[...,1]+=self.eta*self.height*f
        return out
    def radius(self):
        if self.eta==0:return self.base.radius()
        candidates=[]
        for a,b in zip(self.breaks[:-1],self.breaks[1:]):
            grid=np.linspace(np.nextafter(a,b),np.nextafter(b,a),257);k=self.curvature(grid)
            candidates.extend(zip(k,grid))
            for i in range(1,len(grid)-1):
                if k[i]>=k[i-1] and k[i]>=k[i+1]:
                    r=minimize_scalar(lambda u:-float(self.curvature(u)),bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':1e-13});candidates.append((-r.fun,r.x))
        k,u=max(candidates);return float(1/k),float(u)
    def monotone_halves(self):
        left=self.evaluate(np.linspace(self.breaks[0],.5,1001),1)[:,1]
        right=self.evaluate(np.linspace(.5,self.breaks[-1],1001),1)[:,1]
        return bool(np.all(left<=1e-9) and np.all(right>=-1e-9))
