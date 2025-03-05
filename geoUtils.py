#Acknowledgements

#This Source Physics Experiment (SPE) research was funded by the National Nuclear Security Administration, Defense Nuclear Nonproliferation Research and Development (NNSA DNN R&D). The authors acknowledge important interdisciplinary collaboration with scientists and engineers from LANL, LLNL, NNSS, and SNL.

#This Ground-based Nuclear Detonation Detection (GNDD) research was funded by the National Nuclear Security Administration, Defense Nuclear Nonproliferation Research and Development (NNSA DNN R&D).  The authors acknowledge important interdisciplinary collaboration with scientists and engineers from Sandia National Laboratories. 

#Sandia National Laboratories is a multi-mission laboratory managed and operated by National Technology and Engineering Solutions of Sandia, LLC, a wholly owned subsidiary of Honeywell International, Inc., for the U.S. Department of Energy’s National Security Administration under contract DE-NA-0003525.

#Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

import math
from math import cos, sin, atan2, asin, acos, floor, ceil, tan, atan, log, exp, sqrt
import numpy as np

d2r = math.pi / 180.0
Re = 6371.137

# I didn't want to have to do it, but this is the big boy. This returns distance in km, in degrees, and the azimuth as per SAC
# This is close to geo.gps2dist_azimuth(source.lat, source.lon, station[1], station[0])
# Tests show accuracy to the hundredths between the two, but gps2dist_azimuth gives distance in meters, azimuth, and backazimuth
# distaz gives distance in km, great circle arc distance, and azimuth
def distaz(lat1, lon1, lat2, lon2):
    """
    kmdist, gcarc, azimuth = distaz(lat1, lon1, lat2, lon2)
    returns kmdistance, great circle arc degrees, and azimuth between two points
    see also obspy.geodetics.base.gps2dist_azimuth for a similar function
    """
    
    laz = 1
    ldist = 1
    lxdeg = 1
    
    the = lat1
    phe = lon1
    ths = lat2
    phs = lon2
     
    rad = 6378.160
    fl = 0.00335293 # earth flattening
    twopideg = 360.0
    c00 = 1.0
    c01 = 0.25
    c02 = -4.6875e-2
    c03 = 1.953125e-2
    c21 = -0.125
    c22 = 3.125e-2
    c23 = -1.46484375e-2
    c42 = -3.90625e-3
    c43 = 2.9296875e-3

    TODEG = 57.29577950
    TORAD = (1.0 / TODEG)
    FLT_MAX = 64000.0

    ec2 = 2.0 * fl - fl * fl
    onemec2 = 1.0 - ec2
    eps = 1.0 + ec2/onemec2

    temp = the
    if temp == 0.0:
        temp = 1.0e-8

    therad = TORAD * temp
    pherad = TORAD * phe
    
    if (the == 90 or the == -90):
        thg = the*TORAD
    else:
        thg = atan(onemec2 * tan(therad))
    d = sin(pherad)
    e = -cos(pherad)
    f = -cos(thg)
    c = sin(thg)
    a = f*e
    b = -f*d
    g = -c*e
    h = c*d

    temp = ths
    if temp == 0.0:
        temp = 1.0e-8
    thsrad = TORAD*temp
    phsrad = TORAD*phs

    if ths == 90 or ths == -90:
        thg = ths * TORAD
    else:
        thg = atan(onemec2 * tan(thsrad))

    d1 = sin(phsrad)
    e1 = -cos(phsrad)
    f1 = -cos(thg)
    c1 = sin(thg)
    a1 = f1*e1
    b1 = -f1*d1
    g1 = -c1 * e1
    h1 = c1 * d1     
    sc = a * a1 + b * b1 + c * c1
    
    if lxdeg == 1:
        sd = 0.5 * sqrt( (powi(a-a1,2)+powi(b-b1,2)+powi(c-c1,2))*(powi(a+a1,2)+powi(b+b1,2)+powi(c+c1,2)) )
        xdeg = atan2(sd, sc) * TODEG
        if xdeg < 0.0:
            xdeg = xdeg + twopideg
            

    if laz == 1:
        ss = powi(a1 - d, 2) + powi(b1 - e, 2) + powi(c1,2) -2.0
        sc = powi(a1 - g, 2) + powi(b1 - h, 2) + powi(c1 - f, 2) - 2.0
        azi = atan2( ss, sc ) / d2r
        if azi < 0:
            azi = azi + 360.0
            
    if ldist == 1:
        if thsrad > 0.0:
            t1 = thsrad
            p1 = phsrad
            t2 = therad
            p2 = pherad
            if the == 90.0:
                costhk = 0.0
                sinthk = 1.0
                tanthk = FLT_MAX
            elif the == -90.0:
                costhk = 0.0
                sinthk = -1.0
                tanthk = -FLT_MAX
            else:
                costhk = cos(t2)
                sinthk = sin(t2)
                tanthk = sinthk / costhk
            
            if ths == 90.0:
                costhi = 0.0
                sinthi = 1.0
                tanthi = FLT_MAX
            elif ths == -90.0:
                costhi = 0.0
                sinthi = -1.0
                tanthi = -FLT_MAX
            else:
                costhi = cos(t1)
                sinthi = sin(t1)
                tanthi = sinthi / costhi
        else:
            t1 = therad
            p1 = pherad
            t2 = thsrad
            p2 = phsrad
            if ths == 90.0:
                costhk = 0.0
                sinthk = 1.0
                tanthk = FLT_MAX
            elif ths == -90.0:
                costhk = 0.0
                sinthk = -1.0
                tanthk = -FLT_MAX
            else:
                costhk = cos(t2)
                sinthk = sin(t2)
                tanthk = sinthk / costhk
            
            if the == 90.0:
                costhi = 0.0
                sinthi = 1.0
                tanthi = FLT_MAX
            elif the == -90.0:
                costhi = 0.0
                sinthi = -1.0
                tanthi = -FLT_MAX
            else:
                costhi = cos(t1)
                sinthi = sin(t1)
                tanthi = sinthi / costhi
                
        el = ec2/onemec2
        e1 = 1.0 + el
        al = tanthi/(e1*tanthk) + ec2 * sqrt( (e1 + powi(tanthi,2))/(e1+powi(tanthk,2)))
        dl = p1-p2
        a12top = sin(dl)
        a12bot = (al - cos(dl)) * sinthk
        
        a12 = atan2(a12top, a12bot)
        cosa12 = cos(a12)
        sina12 = sin(a12)
        
        e1 = el * (powi(costhk * cosa12,2) + powi(sinthk,2))
        e2 = e1 * e1
        e3 = e1 * e2
        c0 = c00 + c01*e1 + c02*e2 + c03*e3
        c2 = c21*e1 + c22*e2 + c23*e3
        c4 = c42*e2 + c43*e3
        v1 = rad/sqrt(1.0 - ec2*powi(sinthk,2))
        v2 = rad/sqrt(1.0 - ec2*powi(sinthi,2))
        z1 = v1*(1.0 - ec2) * sinthk
        z2 = v2*(1.0 - ec2) * sinthi
        x2 = v2 * costhi * cos(dl)
        y2 = v2 * costhi * sin(dl)
        e1p1 = e1 + 1.0
        sqrte1p1 = sqrt(e1p1)
        u1bot = sqrte1p1*cosa12
        u1 = atan2(tanthk, u1bot)
        u2top = v1 * sinthk + e1p1 * (z2-z1)
        u2bot = sqrte1p1 * (x2*cosa12 - y2*sinthk*sina12)
        u2 = atan2(u2top, u2bot)
        b0 = v1*sqrt(1.0 + el*powi(costhk*cosa12,2))/e1p1
        du = u2-u1
        pdist = b0*(c2*(sin(2.0*u2)-sin(2.0*u1))+c4*(sin(4.0*u2)-sin(4*u1)))
        dist = abs(b0*c0*du+pdist)
    return dist, xdeg, azi
        


# utm2deg and deg2utm are transcribed by Rob Porritt on 12.2.2020 from codes packaged with Utah Preliminary model from Charles Hoots and Leiph Preston.
# These utilities are based on Matlab codes from 2006 by Rafael Palacios from Universidad Pontificia Comillas, Marid, Spain
def deg2utm(lat, lon):
    """
    x, y, utmzone = deg2utm(lat,lon)
    given a latitude and longitude, finds utm zone and easting and northing
    """
    n1 = lat.size
    n2 = lon.size
    assert n1 == n2, "lat and lon vectors must be the same length"
    x = np.zeros((n1,1))
    y = np.zeros((n2,1))
    utmzone = []
    
    sa = 6378137.0
    sb = 6356752.314245
    e2 = (((sa**2.0)-(sb**2.0))**0.5)/sb
    e2cuadrada = e2 ** 2.0
    c = (sa ** 2) / sb;
    
    for i in range(n1):
        la = lat[i] * d2r
        lo = lon[i] * d2r
        
        Huso = int((lon[i] / 6) + 31)
        S = ((Huso * 6) - 183)
        deltaS = lo - ( S*d2r)
        
        if lat[i]<-72:
            Letra='C'
        elif lat[i]<-64:
            Letra='D'
        elif lat[i]<-56:
            Letra='E'
        elif lat[i]<-48: 
            Letra='F'
        elif lat[i]<-40:
            Letra='G'
        elif lat[i]<-32:
            Letra='H'
        elif lat[i]<-24: 
            Letra='J'
        elif lat[i]<-16: 
            Letra='K'
        elif lat[i]<-8: 
            Letra='L'
        elif lat[i]<0: 
            Letra='M'
        elif lat[i]<8:
            Letra='N'
        elif lat[i]<16:
            Letra='P'
        elif lat[i]<24:
            Letra='Q'
        elif lat[i]<32:
            Letra='R'
        elif lat[i]<40:
            Letra='S'
        elif lat[i]<48:
            Letra='T'
        elif lat[i]<56:
            Letra='U'
        elif lat[i]<64:
            Letra='V'
        elif lat[i]<72:
            Letra='W'
        else:
            Letra='X';
            
        a = cos(la) * sin(deltaS)
        epsilon = 0.5 * log((1.0+a)/(1.0-a))
        nu = atan(tan(la)/cos(deltaS)) - la
        v = ( c / ( ( 1.0 + ( e2cuadrada * ( cos(la) ) ** 2.0 ) ) ) ** 0.5 ) * 0.9996;
        ta = ( e2cuadrada / 2.0 ) * epsilon ** 2.0 * ( cos(la) ) ** 2.0;
        a1 = sin( 2.0 * la );
        a2 = a1 * ( cos(la) ) ** 2.0;
        j2 = la + ( a1 / 2.0 );
        j4 = ( ( 3.0 * j2 ) + a2 ) / 4.0;
        j6 = ( ( 5.0 * j4 ) + ( a2 * ( cos(la) ) ** 2.0) ) / 3.0;
        alfa = ( 3.0 / 4.0 ) * e2cuadrada;
        beta = ( 5.0 / 3.0 ) * alfa ** 2.0;
        gama = ( 35.0 / 27.0 ) * alfa ** 3.0;
        Bm = 0.9996 * c * ( la - alfa * j2 + beta * j4 - gama * j6 );
        xx = epsilon * v * ( 1.0 + ( ta / 3.0 ) ) + 500000.0;
        yy = nu * v * ( 1.0 + ta ) + Bm;

        if yy<0.0:
            yy=9999999.0+yy;

        x[i]=xx;
        y[i]=yy;
        
        utmzone.append('{:02d} {}'.format(Huso,Letra));

    return x, y, utmzone


def utm2deg(xx, yy, utmzone):
    """
    lat, lon = utm2deg(xx,yy, utmzone)
    given xx and yy as meters in utmzone, returns latitude and longitude    
    """
    n1 = xx.size
    n2 = yy.size
    n3 = len(utmzone)
    
    assert n1 == n2 and n1 == n3, "x and y vectors must be the same length as the utmzone list"
    
    Lat = np.zeros((n1,1))
    Lon = np.zeros((n1,1))
    
    sa = 6378137.0
    sb = 6356752.314245
    e2 = (((sa**2.0)-(sb**2.0))**0.5)/sb
    e2cuadrada = e2 ** 2.0
    c = (sa ** 2) / sb;
    
    for i in range(n1):
        if isinstance(utmzone[i],str):
            regionCode = utmzone[i].split()
        else:
            regionCode = utmzone[i][0].split()
        assert regionCode[1] <= "X" and regionCode[1] >= "C", "Error, utm zone should be a vector of strings like 30 T"
        if regionCode[1] > "M":
            hemis = "N"
        else:
            hemis = "S"
            
        x = xx[i]
        y = yy[i]
        zone = float(regionCode[0])
        
        X = x-500000
        
        if hemis == 'S' or hemis == 's':
            Y = y - 10000000.0;
        else:
            Y = y;
    
        S = ( ( zone * 6.0 ) - 183.0 )
        lat =  Y / ( 6366197.724 * 0.9996 )
        v = ( c / ( ( 1.0 + ( e2cuadrada * ( cos(lat) ) ** 2 ) ) ) ** 0.5 ) * 0.9996
        a = X / v
        a1 = sin( 2.0 * lat )
        a2 = a1 * ( cos(lat) ) ** 2.0
        j2 = lat + ( a1 / 2.0 )
        j4 = ( ( 3.0 * j2 ) + a2 ) / 4.0
        j6 = ( ( 5.0 * j4 ) + ( a2 * ( cos(lat) ) ** 2.0) ) / 3.0
        alfa = ( 3.0 / 4.0 ) * e2cuadrada
        beta = ( 5.0 / 3.0 ) * alfa ** 2.0
        gama = ( 35.0 / 27.0 ) * alfa ** 3.0
        Bm = 0.9996 * c * ( lat - alfa * j2 + beta * j4 - gama * j6 )
        b = ( Y - Bm ) / v
        Epsi = ( ( e2cuadrada * a ** 2.0 ) / 2.0 ) * ( cos(lat) ) ** 2.0
        Eps = a * ( 1.0 - ( Epsi / 3.0 ) )
        nab = ( b * ( 1.0 - Epsi ) ) + lat
        senoheps = ( exp(Eps) - exp(-Eps) ) / 2.0
        Delt = atan(senoheps / (cos(nab) ) )
        TaO = atan(cos(Delt) * tan(nab))
        longitude = (Delt *(180.0 / math.pi ) ) + S
        latitude = ( lat + ( 1.0 + e2cuadrada* (cos(lat) ** 2.0) - ( 3.0 / 2.0 ) * e2cuadrada * sin(lat) * cos(lat) * ( TaO - lat ) ) * ( TaO - lat ) ) * (180.0 / math.pi)
   
        Lat[i]=latitude
        Lon[i]=longitude
   
    return Lat, Lon



# powi and computeAzimuth modified from seismic analysis code (SAC) by Rob Porritt 12.1.2020
def powi(b, x):
    """
    p = powi(b,x)
    calculate b to the power of x
    """
    temp = 0.0
    if (b  == 0.0):
        return 0.0
    if (x == 0):
        return 1.0
  
    if (x > 0):
        temp = b
        for i in range(x-1,0,-1):
            temp *= b
        return temp
    if (x < 0):
        temp = 1.0 / b
        for i in range(x+1, 0):
            temp *= (1.0/b)
        return temp
    return temp
  


def computeAzimuth(lat1, lon1, lat2, lon2):
    """
    azi = computeAzimuth(lat1, lon1, lat2, lon2)
    returns azimuth between two points
    see also obspy.geodetics.base.gps2dist_azimuth for a similar function
    """
    
    the = lat1
    phe = lon1
    ths = lat2
    phs = lon2
     
    rad = 6378.160
    fl = 0.00335293 # earth flattening
    twopideg = 360.0
    c00 = 1.0
    c01 = 0.25
    c02 = -4.6875e-2
    c03 = 1.953125e-2
    c21 = -0.125
    c22 = 3.125e-2
    c23 = -1.46484375e-2
    c42 = -3.90625e-3
    c43 = 2.9296875e-3

    TODEG = 57.29577950
    TORAD = (1.0 / TODEG)

    ec2 = 2.0 * fl - fl * fl
    onemec2 = 1.0 - ec2
    eps = 1.0 + ec2/onemec2

    temp = the
    if temp == 0.0:
        temp = 1.0e-8

    therad = TORAD * temp
    pherad = TORAD * phe
    
    if (the == 90 or the == -90):
        thg = the*TORAD
    else:
        thg = atan(onemec2 * tan(therad))
    d = sin(pherad)
    e = -cos(pherad)
    f = -cos(thg)
    c = sin(thg)
    a = f*e
    b = -f*d
    g = -c*e
    h = c*d

    temp = ths
    if temp == 0.0:
        temp = 1.0e-8
    thsrad = TORAD*temp
    phsrad = TORAD*phs

    if ths == 90 or ths == -90:
        thg = ths * TORAD
    else:
        thg = atan(onemec2 * tan(thsrad))

    d1 = sin(phsrad)
    e1 = -cos(phsrad)
    f1 = -cos(thg)
    c1 = sin(thg)
    a1 = f1*e1
    b1 = -f1*d1
    g1 = -c1 * e1
    h1 = c1 * d1     
    sc = a * a1 + b * b1 + c * c1

    ss = powi(a1 - d, 2) + powi(b1 - e, 2) + powi(c1,2) -2.0
    sc = powi(a1 - g, 2) + powi(b1 - h, 2) + powi(c1 - f, 2) - 2.0
    azi = atan2( ss, sc ) / d2r
    if azi < 0:
        azi = azi + 360.0
    return azi


def getProfilePoints(point1, point2, delta, beginPad=0.0, endPad=0.0, multipleOf=1, returnAzimuth=False,
                     azimuthAccuracy=""):
    # Rotate point2 through a rotation that would put point1 at the North Pole:
    newPoint2 = rotatePoint(point2, (0, point1[1] + 90.0), point1[0] - 90.0)
    azimuth = 180 - newPoint2[1] + point1[1]
    if azimuth > 180:
        azimuth -= 360
    if azimuth < - 180:
        azimuth += 360
    colats = np.arange(-beginPad, (90 - newPoint2[0]) * d2r * Re + endPad + delta, delta)
    if len(colats) % multipleOf != 0:
        extraNeeded = multipleOf - len(colats) % multipleOf
        beginExtra = floor(extraNeeded / 2)
        endExtra = ceil(extraNeeded / 2)
        colats = np.arange(-beginPad - beginExtra * delta,
                           (90 - newPoint2[0]) * d2r * Re + endPad + delta + endExtra * delta, delta)
        assert (len(colats) % multipleOf == 0)
    colats = (1.0 / (d2r * Re)) * colats

    lons = np.ones_like(colats) * newPoint2[1]
    lons[colats < 0] = lons[colats < 0] + 180.0
    colats[colats < 0] = -1 * colats[colats < 0]
    lats = 90.0 - colats
    output = []
    for lat, lon in zip(lats, lons):
        output.append(rotatePoint((lat, lon), (0, point1[1] + 90.0), 90.0 - point1[0]))
    if returnAzimuth is True:
        return (output, azimuth)
    else:
        return output


def rotatePoints(points, axis, angle):
    for point in points:
        point = rotatePoint(point, axis, angle)
    return points


def rotatePoint(point, axis, angle):
    vaxis = ll2vec(axis)
    vpoint = ll2vec(point)
    rangle = angle * d2r
    dprod = vaxis[0] * vpoint[0] + vaxis[1] * vpoint[1] + vaxis[2] * vpoint[2]
    rvpoint = (cos(rangle) * vpoint[0] + sin(rangle) * (vaxis[1] * vpoint[2] - vaxis[2] * vpoint[1])
               + dprod * (1 - cos(rangle)) * vaxis[0],
               cos(rangle) * vpoint[1] + sin(rangle) * (vaxis[2] * vpoint[0] - vaxis[0] * vpoint[2])
               + dprod * (1 - cos(rangle)) * vaxis[1],
               cos(rangle) * vpoint[2] + sin(rangle) * (vaxis[0] * vpoint[1] - vaxis[1] * vpoint[0])
               + dprod * (1 - cos(rangle)) * vaxis[2])
    return vec2ll(rvpoint)


def ll2vec(point):
    lat = point[0]
    lon = point[1]
    out = (cos(d2r * lat) * cos(d2r * lon), cos(d2r * lat) * sin(d2r * lon), sin(d2r * lat))
    return out


def vec2ll(vec):
    lat = asin(vec[2]) / d2r
    lon = atan2(vec[1], vec[0]) / d2r
    out = (lat, lon)
    return out


def distance(point1, point2):
    v1 = ll2vec(point1)
    v2 = ll2vec(point2)
    alpha = acos(v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2])
    return alpha * Re

def calcCoord(point1,point2,distance):
    newPoint2 = rotatePoint(point2, (0, point1[1] + 90.0), point1[0] - 90.0)
    returnPoint = (90-newPoint2[0]+distance/Re/d2r,newPoint2[1])
    returnPoint = rotatePoint(returnPoint, (0, point1[1] + 90.0), 90.0 - point1[0])
    return returnPoint
