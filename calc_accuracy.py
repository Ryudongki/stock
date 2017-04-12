with open("7.txt", "r") as f:
	cnt = 0
	zero = 0
	for line in f.read().split('\n'):
		try:
			temp = int(line)
			cnt += 1
			if temp == 0:
				zero += 1
		except Exception as e:
			pass
print(1.0 * zero / cnt)


