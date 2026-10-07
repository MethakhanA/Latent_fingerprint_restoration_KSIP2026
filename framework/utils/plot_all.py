import matplotlib.pyplot as plt
import math
def plot_all(img_list,  rows:int=None, cols:int=None,title_list=None, figsize=10, cmap='gray', normalization=False, fullscreen=False):
    '''
        plot all images in the list. Using title and cmap as parameters
    '''
    if type(img_list) is not list:
        # work around. Turn it into list
        img_list = [img_list]
    if rows is None:
        rows = 1
        cols = len(img_list)
    elif cols is None:
        cols = math.ceil(len(img_list)/rows)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(figsize, figsize))
    fig.tight_layout()
    if fullscreen:
        fig.canvas.manager.full_screen_toggle()
    for img in range(len(img_list)):
        plt.subplot(rows, cols, img+1)
        if title_list is not None:
            # If there is a title list
            plt.title(title_list[img])
        if cmap is None:
            if not normalization:
                plt.imshow(img_list[img])
            else:
                plt.imshow(img_list[img], vmax=255, vmin=0)
        else:
            if type(cmap) is list:
                if not normalization:
                    plt.imshow(img_list[img],cmap=cmap[img])
                else:
                    plt.imshow(img_list[img],cmap=cmap[img], vmax=255, vmin=0)
            else:
                if not normalization:
                    plt.imshow(img_list[img], cmap=cmap)
                else:
                    plt.imshow(img_list[img], cmap=cmap, vmax=255, vmin=0)
    plt.show()