def dicho_search_nearest ( L, v ):
    
    if L[0]>v : return 0
    l,r=0,len(L)-1
    while l<r:
        m=(l+r)//2
        if L[m]<v: l=m+1
        else :     r = m
    
    return L[l]


def convolution ( im, ker, borders=0 ):

    padding = (np.array(ker.shape)-1)
    N,M = padding//2

    #smaller image to apply the kernel to (no copying)
    temp = np.zeros(im.shape)
    for i in range(N,im.shape[0]-N-1):
        for j in range(M,im.shape[1]-M-1):
                temp[i,j]=np.sum(ker*im[i:i+2*N+1,j:j+2*M+1])
    return temp

    #np.pad !!!!    

    '''
    #larger image to apply the kernel to
    temp = borders*np.ones(np.shape(im)+padding)
    temp[N:-N,M:-M]=im
    return np.array([ [ np.sum(ker*temp[ i:i+2*N+1, j:j+2*M+1]) for j in range(np.shape(im)[1]) ] for i in range(np.shape(im)[0]) ])
    '''


