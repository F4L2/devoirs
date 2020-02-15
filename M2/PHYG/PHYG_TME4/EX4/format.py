out = open("input.trees","w")
with open("raw.trees", 'r') as f:
	string = ""
	for line in f:
		line = line.strip()
		string += line
		if(";" in line):
			out.write(string)
			out.write("\n")
			string = ""
		
