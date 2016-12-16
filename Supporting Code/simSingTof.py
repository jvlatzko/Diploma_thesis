''' script for generating Luxrender lxs files for ToF Simulation'''
''' Stephan Meister <stephan.meister@iwr.uni-heidelberg.de> 
	Heidelberg Collaboratory for Image Processing (HCI)
	Interdisciplinary Institute for Scientific Computing (IWR)
	University of Heidelberg, 2013'''


import re
import sys	
import math
import os
import subprocess
import h5py
import time

def s2Int(s):
	if s[-1] in ["M","m"]:
		return int(s[:-1]) * 1000000
	if s[-1] in ["K","k"]:
		return int(s[:-1]) * 1000
	return int(s)

def int2S(i):
	if i // 1000000 > 0:
		return "%uM" % (i // 1000000)
	if i // 1000 > 0:
		return "%uk" % (i // 1000)
	return "%u" % i

def getVal(default):
	r = input().strip()
	if r == '':
		return default
	else:
		return r
	
def replaceOrInsert(value, section, target):
	if re.search(value,target):
		return target
	else:
		target = re.sub(section, '''%s\n\t%s''' % (section, value),target)
	return target

luxconsole = "/home/jan/ToF-Simulator/HCI_TofSimulator_latest/src/lux/luxconsole"
results_outdir = "results"

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("No template lxs provided!")
		exit()
	f = open(sys.argv[1])
	template = f.read()
	f.close()
	
	# --------------------------------- lxs file template ------------------------
	
	#print("Eye depth [6]: ", end="")
#	eyeDepth = s2Int(getVal("6"))
	eyeDepth = 8
	#print("Light depth [6]: ", end="")
#	lightDepth = s2Int(getVal("6"))
	lightDepth = 8
	print("Samples [1k]: ",end="")
	samples = s2Int(getVal("1k"))
#	samples = 1000
	#print("ToF Freq [20M]: ",end="")
#	toffreq = s2Int(getVal("20M"))
	toffreq = 20000000
	#print("Harmonics [0]: ", end="")
#	harmonics = s2Int(getVal("0"))
	harmonics = 0
	#print("Harmonics shift[0.0]: ", end="")
#	harmonicsShift = (getVal("0.0"))
	harmonicsShift = 0.0
	#print("Harmonics Intensity[0.0]: ", end="")
#	harmonicsInt = (getVal("0.0"))
	harmonicsInt = 0.0

	#print("Render output directory [render]: ",end="")
	#render_outdir = getVal("render").replace('''\\''','''/''')
	render_outdir = "render"
	#print("lxs file prefix [render]: ",end="")
	#file_prefix = getVal("render")
	file_prefix = "render"
#	print("cameramatrix file [../../tof_internal.txt]:",end="")
	cam_matrix = "/home/jan/ToF-Simulator/tof_internal_deg.txt"
#	cam_matrix = getVal("../../tof_internal.txt")
	print("comments []",end="")
	comments = getVal("")
	
	if not render_outdir == "" and not os.path.exists(render_outdir):
		os.makedirs(render_outdir)
	
	p = re.compile(r'''eyedepth" \[[0-9]+\]''')
	template = re.sub(p,'''eyedepth" [%u]''' % eyeDepth,template,count = 1)
	p = re.compile(r'''lightdepth" \[[0-9]+\]''')
	template = re.sub(p,'''lightdepth" [%u]''' % lightDepth,template,count = 1)
	p = re.compile(r'''haltspp" \[[0-9]+\]''')
	template = re.sub(p,'''haltspp" [%u]''' % samples,template,count = 1)

	p = re.compile('''"integer displayinterval" \[[0-9]+\]''')
	template = re.sub(p,'''"integer displayinterval" [60]''',template,count = 1)
	p = re.compile('''"integer writeinterval" \[[0-9]+\]''')
	template = re.sub(p,'''"integer writeinterval" [60]''',template,count = 1)
	p = re.compile('''"integer flmwriteinterval" \[[0-9]+\]''')
	template = re.sub(p,'''"integer flmwriteinterval" [120]''',template,count = 1)

	template = replaceOrInsert('''"float toffreq" [%s]''' % toffreq,'''SurfaceIntegrator "bidirectional"''',template)
	template = replaceOrInsert('''"float modint" [0.75]''','''SurfaceIntegrator "bidirectional"''',template)
	template = replaceOrInsert('''"float modoffset" [0.25]''','''SurfaceIntegrator "bidirectional"''',template)
	template = replaceOrInsert('''"float phaseshift" [0]''','''SurfaceIntegrator "bidirectional"''',template)
	template = replaceOrInsert('''"integer harmonics" [%s]'''% harmonics,'''SurfaceIntegrator "bidirectional"''',template)
	template = replaceOrInsert('''"float harmonicsint" [%s]'''% harmonicsInt,'''SurfaceIntegrator "bidirectional"''',template)
	template = replaceOrInsert('''"float harmonicsshift" [%s]'''% harmonicsShift,'''SurfaceIntegrator "bidirectional"''',template)
	
	
	phaseshifts = [0, 0.5 * math.pi, 1.0 * math.pi, 1.5 * math.pi]

	lxsFiles = []
	exr_files = []
	for i in [0,1,2,3]:
		p = re.compile(r'''filename" \[".*?"\]''')
		exr_filepath = "%s/%s_%seyeDepth_%slightDepth_%sSamples_%u" % (render_outdir, file_prefix,int2S(eyeDepth),int2S(lightDepth),int2S(samples),i)
		exr_files.append(exr_filepath+".exr")
		template = re.sub(p,'''filename" ["%s"]''' %(exr_filepath), template,count = 1)
		
		p = re.compile(r'''phaseshift" \[.*?]''')
		template = re.sub(p,'''phaseshift" [%f]''' % phaseshifts[i],template,count = 1)
		
		filename = "%s_%seyeDepth_%slightDepth_%sSamples_%u.lxs" % (file_prefix, int2S(eyeDepth), int2S(lightDepth), int2S(samples),i)
		out = open(filename,"w")
		out.write(template)
		out.close()
		lxsFiles.append(filename)
		print(lxsFiles[i])
	if len(sys.argv) > 2 and sys.argv[2] == "--norun":
		exit()
	
	#----------------------------- Run Renderer -----------------------
	
	for i  in [0,1,2,3]:
		#pass
		print("\n\nRunning Luxrender on file %i\n" % i)
		p = subprocess.Popen([luxconsole, lxsFiles[i]])
		p.wait()
	
	print("\nfinished")
		
	