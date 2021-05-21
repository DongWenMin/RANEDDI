from multiprocessing.pool import Pool
# a = [1,2,3,4]
# b = [4,5,6,7]
# zipped = zip(a,b)
# for item in zipped:
#     print(item)

def muti(x,y):
    return 2*x
if __name__ == "__main__":
    pool = Pool(processes=2)
    result = pool.map(muti,[(5,6),(8,9),(9,1)])
    print(result)