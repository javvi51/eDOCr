from anytree import RenderTree, NodeMixin
import gc
import math
import numpy as np
import cv2
from shapely.geometry import box
###################---Code Structure---######################################################################
#
#Classes
# |
# |-> rect(NodeMixin)
#
#############################################################################################################
#
#Functions
# |-> find_em(classType, attr, targ) -> Given a class type, a known attribute and its value, it can return the instance it refers to
# |-> findrect(img) -> Returns a list with rect classes and img_boxes with rectangles plot
# |   |-> box_tree(rect_list,class_list) -> Build the rectangle tree
# |   |   |-> complete_level(rect_list,class_list) -> Complete only the top level: identifies the parents and their children
# |   |       |-> biggest(rect_list) -> returns the biggest box in the list
# |   |       |-> get_children(rect_list,parent_name) -> returns the all progeny in the list for a given parent
# |   |-> show_tree(rect_list) -> OPTIONAL, show the box tree structure           
# |   |-> angle(pt0,pt1,pt2) -> get the angle of the lines conformed by pt0-pt1 and pt0-pt2
class rect(NodeMixin):
    def __init__(self, name, x, y ,w, h, crop_img,state, parent=None, children=None):
        super(rect, self).__init__()
        self.name = name
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.crop_img =crop_img
        self.state = state
        self.parent = parent
        if children:
            self.children = children

def find_em(classType, attr, targ):
    '''Given a class type, a known attribute and its value, 
    it can return the instance it refers to.
    Args:
        class type: class type
        attr: attribute in class type
        targ: value of the attribute
    Returns:
        class instance
        '''
    return [obj for obj in gc.get_objects() if isinstance(obj, classType) and getattr(obj, attr)==targ]

def biggest(rect_list):
    '''Gets the biggest rect (class) in the list
    Args: rect_list: list of rects
    Returns: max_name: rect name
     '''
    sizes = (rect_list[:,3]*rect_list[:,4]).astype(float)
    max_size = np.amax(sizes)
    max_index=np.where(sizes==max_size)[0][0]
    max_name=rect_list[max_index,0]
    return max_name

def get_children(rect_list, parent_name):
    '''returns all the progeny in the list for a given parent.
    Args:
        rect_list: list of rects
        parent_name: parent name
    Returns: children_names: list of names of rects chldren from parent
        '''
    parent_index=np.where(rect_list[:,0]==parent_name)[0][0]
    box_array=[]
    for i in range(0,len(rect_list)):
        minx=rect_list[i,1].astype(float)
        miny=rect_list[i,2].astype(float)
        maxx=rect_list[i,1].astype(float)+rect_list[i,3].astype(float)
        maxy=rect_list[i,2].astype(float)+rect_list[i,4].astype(float)
        box_array.append([rect_list[i,0],box(minx,miny,maxx,maxy)])
    parent_array=box_array.pop(parent_index)
    children_names=[]
    for i in range(0,len(box_array)):
        if box_array[i][1].covered_by(parent_array[1])==True:
            children_names.append(box_array[i][0])
        else:
            pass
    return children_names


def complete_level(rect_list):
    '''Completes only the top level: identifies the parents and their children
    Args:
        rect_list: list of rects
    Returns: parents: list of names of rect parents
    '''
    parents=[]
    childrens=[]
    while np.size(rect_list):
        #Parent stuff
        parent_name=biggest(rect_list)
        parents.append(parent_name)
        parent_index=np.where(rect_list[:,0]==parent_name)[0][0]
        
        #children stuff
        children=get_children(rect_list,parent_name)
        rect_list=np.delete(rect_list, parent_index, 0)
        for i in range(0,len(children)):
            children_index=np.where(rect_list[:,0]==children[i])[0][0]
            rect_list=np.delete(rect_list, children_index, 0)
            childrens.append(children[i])
            find_em(rect, 'name', children[i])[0].parent=find_em(rect, 'name', parent_name)[0]
            
    return parents

def box_tree(rect_list):
    '''Builds the rect tree of parents and children
    Args: rect_list: list of rects '''
    rect_list=np.array(rect_list)
    while np.size(rect_list):  #While the list has rectangles
        parents=complete_level(rect_list) #Get parents and children for every level
        for i in range(0,len(parents)): #Pick only parents
            parent_index=np.where(rect_list[:,0]==parents[i])[0][0] 
            rect_list=np.delete(rect_list, parent_index, 0) #Delete parents

def angle(pt0,pt1,pt2):
    '''get the angle of the lines conformed by pt0-pt1 and pt0-pt2
    Args: 
        pt0: np array with x,y coordinates
        pt1: np array with x,y coordinates
        pt2: np array with x,y coordinates
    Returns:
        angle: angle conformed by the lines, in degrees
        '''
    dx1 = pt1[0][0] - pt0[0][0]
    dy1 = pt1[0][1] - pt0[0][1]
    d1 = np.array([dx1, dy1])
    dx2 = pt2[0][0] - pt0[0][0]
    dy2 = pt2[0][1]  - pt0[0][1]
    d2 = np.array([dx2, dy2])

    dotPr = np.dot(d1, d2)

    lengthOfd1 = math.sqrt(math.pow(dx1,2) + math.pow(dy1, 2))
    lengthOfd2 = math.sqrt(math.pow(dx2, 2) + math.pow(dy2, 2))

    alfa=np.arccos(dotPr/(lengthOfd1*lengthOfd2))
    alfa=alfa*180/math.pi
    return alfa

def findrect(img):
    '''Returns a list with rect classes from an image
    Args: img: the image (mechanical engineering drawing)
    returns: 
        class_list: list of rects
        img_boxes: image with the contourns of the boxes'''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Convert img to grayscale
    _,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV) #Threshold to binary image
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours = cv2.findContours(dilated,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0] #Get contourns
    
    img_boxes=img.copy()
    r=0
    rect_list=[]
    class_list=[]
    for cnt in contours:
        x1,y1 = cnt[0][0] #Top-left corner
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True) #Approach cnt to polygons with multiple points
        if len(approx) == 4 and 88<angle(approx[1],approx[0],approx[2])<92 and 88<angle(approx[3],approx[0],approx[2])<92: #if cnt can be approx with only 4 points, it is a 4 side polygon
            x, y, w, h = cv2.boundingRect(cnt) #get rectangle information
            if w*h>1000: #Clean very small rectangles
                crop_img=img[y:y+h,x:x+w] #Crop the rectangle
                cv2.putText(img_boxes, 'rect_'+str(r), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 127, 83), 2) #Add rectangle tag
                img_boxes = cv2.drawContours(img_boxes, [cnt], -1, (255, 127, 83), 2) #Plot rectangle contourn in green
                rect_list.append(['rect_'+str(r),x,y,w,h]) #Get a list of rectangles
                class_list.append(rect('rect_'+str(r),x,y,w,h,crop_img,'green')) #Get a list with the rect class
                r=r+1
    
    print('number of rectangles:',len(rect_list))
    box_tree(rect_list)  #Build the rectangles tree, parents and children
    #show_tree(rect_list) #Optional: show the rectangles tree
    return class_list, img_boxes

def show_tree(rect_list):
    '''prints the box tree structure
    Args: rect_list: list of rects'''
    bigdaddy=find_em(rect, 'name', biggest(np.array(rect_list)))[0]
    for pre, _, node in RenderTree(bigdaddy):
        treestr = u"%s%s" % (pre, node.name)
        print(treestr.ljust(8), node.size)