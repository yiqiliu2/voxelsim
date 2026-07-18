import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
plt.rc("hatch", color="white")  # Set default hatch color to white
# class SplitLegendBox(HandlerPatch):
#     def create_artists(self, legend, orig_handle,
#                        xdescent, ydescent, width, height, fontsize, trans):
#         # Create two rectangles, each half the width, side by side
#         w = width / 2
#         h = height
#         patch1 = mpatches.Rectangle([xdescent, ydescent], w, h, facecolor=orig_handle.colors[0], edgecolor='black', transform=trans)
#         patch2 = mpatches.Rectangle([xdescent + w, ydescent], w, h, facecolor=orig_handle.colors[1], edgecolor='black', transform=trans)
#         return [patch1, patch2]
class SplitLegendBox(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # print(f"hatches = {getattr(orig_handle, 'hatches', None)}")
        hatches = getattr(orig_handle, 'hatches', None)
        num_colors = len(orig_handle.colors)
        
        if num_colors == 2:
            # Two triangles splitting diagonally
            triangle1 = mpatches.Polygon(
                [[xdescent, ydescent + height],
                 [xdescent, ydescent],
                 [xdescent + width, ydescent]],
                facecolor=orig_handle.colors[0],
                edgecolor='black',
                hatch=hatches[0] if hatches else None,
                transform=trans
            )
            triangle2 = mpatches.Polygon(
                [[xdescent, ydescent + height],
                 [xdescent + width, ydescent + height],
                 [xdescent + width, ydescent]],
                facecolor=orig_handle.colors[1],
                edgecolor='black',
                hatch=hatches[1] if hatches else None,
                transform=trans
            )
            triangle1._hatch_color = mpl.colors.to_rgba('white')
            triangle2._hatch_color = mpl.colors.to_rgba('white')
            return [triangle1, triangle2]
        
        elif num_colors == 3:
            # Three vertical rectangles
            w = width / 3
            h = height
            
            # Left rectangle
            rect1 = mpatches.Rectangle(
                [xdescent, ydescent], w, h,
                facecolor=orig_handle.colors[0],
                edgecolor='black',
                hatch=hatches[0] if hatches else None,
                transform=trans
            )
            # Middle rectangle
            rect2 = mpatches.Rectangle(
                [xdescent + w, ydescent], w, h,
                facecolor=orig_handle.colors[1],
                edgecolor='black',
                hatch=hatches[1] if hatches else None,
                transform=trans
            )
            # Right rectangle
            rect3 = mpatches.Rectangle(
                [xdescent + 2*w, ydescent], w, h,
                facecolor=orig_handle.colors[2],
                edgecolor='black',
                hatch=hatches[2] if hatches else None,
                transform=trans
            )
            rect1._hatch_color = mpl.colors.to_rgba('white')
            rect2._hatch_color = mpl.colors.to_rgba('white')
            rect3._hatch_color = mpl.colors.to_rgba('white')
            return [rect1, rect2, rect3]
        
        else:
            raise ValueError(f"SplitLegendBox supports 2 or 3 colors, got {num_colors}")
    

class SplitPatch:
    def __init__(self, colors, hatches=None):
        self.colors = colors
        self.hatches = hatches if hatches is not None else [None] * len(colors)

# Example usage
if __name__ == "__main__":
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    
    # Example with 2 colors
    ax[0].bar([0], [1], color='red')
    ax[0].bar([1], [1], color='blue')
    split_patch_2 = SplitPatch(colors=['red', 'blue'], hatches=['/', '\\'])
    ax[0].legend([split_patch_2], ['2 Colors'], handler_map={SplitPatch: SplitLegendBox()}, ncol=1, handleheight=1.5, handlelength=1.5)
    ax[0].set_title('2-Color Split Legend')
    
    # Example with 3 colors
    ax[1].bar([0], [1], color='red')
    ax[1].bar([1], [1], color='green')
    ax[1].bar([2], [1], color='blue')
    split_patch_3 = SplitPatch(colors=['red', 'green', 'blue'], hatches=['/', '\\', 'x'])
    ax[1].legend([split_patch_3], ['3 Colors'], handler_map={SplitPatch: SplitLegendBox()}, ncol=1, handleheight=1.5, handlelength=1.5)
    ax[1].set_title('3-Color Split Legend')
    
    plt.tight_layout()
    plt.savefig("figures/split_legend_example.pdf")