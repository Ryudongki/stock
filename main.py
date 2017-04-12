import os
def main():
    with open("accuracy.txt", 'w') as f:
        pass

    file_list = []

    for (path, dir, files) in os.walk(os.path.join(os.getcwd(), 'data')):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if (ext == '.csv'):
                file_list.append(filename)

    for file in file_list:
        print(os.path.splitext(file)[0])
        os.system('python Predict.py {}'.format(os.path.splitext(file)[0]))

if __name__ == "__main__":
    main()