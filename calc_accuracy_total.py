with open("accuracy.txt", "r") as f:
    cnt = 0
    sum = 0
    for line in f.read().split('\n'):
        try:
            sum += float(line.lstrip('[').rstrip(']'))
            cnt += 1
        except Exception as e:
            pass
print(sum / cnt)