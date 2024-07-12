
import sys 
sys.path.append('/home/nxy/codes/coil_spline_HTS/HTS')
import material_jcrit




j,b,t = material_jcrit.get_critical_current(4.2, 16, 0.005,'REBCO' )
print(j,b,t)


I = 12163433.12

sec = I / j * 1e4
width = sec**0.5
print(sec, width)

