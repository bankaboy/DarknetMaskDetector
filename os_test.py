from os import walk

# f = []
# for (dirpath, dirnames, filenames) in walk("./people_without_masks/"):
#     f.extend(filenames)
#     break
# print(f)

count = 0
_,_,filenames = next(walk("./people_without_masks/"))
for file in filenames:
    print(file)
    # count += 1
# print(count)


# print(walk("./people_without_masks/"))