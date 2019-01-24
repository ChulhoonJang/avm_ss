# loading packages
# - public lib
import os, imp, json, sys, time, cv2, scipy.misc
import numpy as np
import tensorflow as tf

class avm_ss: # semantic segmentation for avm image    
    def __init__(self, filepath):
        self.graph = self.load_pb(filepath)
        self.output = self.graph.get_tensor_by_name('decoder/Softmax:0')
        self.input = self.graph.get_tensor_by_name('Placeholder:0')    
        self.sess = tf.Session(graph=self.graph)

        hypes_filepath = './models/hypes.json'
        with open(hypes_filepath, 'r') as f:
            self.hypes = json.load(f)        

        self.color_seg={1:(0,153,255,127), 2:(0,255,0,127), 3:(255,0,0,127), 4:(255,0,255,127)}
                
        self.num_classes = self.hypes['num_classes']
        self.threshold=[]
        for c in range(self.num_classes):
            threshold_name = 'threshold_{:d}'.format(c)
            threshold = np.array(self.hypes['threshold'][threshold_name])
            self.threshold.append(threshold)
        
        print('[avm_ss] model graphs are built')
    
    def load_pb(self, pb):
        with tf.gfile.GFile(pb, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
            
        return graph  
        

# Core() is called everytime you have a new input
    def run(self, img_src):
        if img_src is not None:

            ################################################
            # inference
            start_time = time.time()
            output = self.sess.run(self.output, {self.input: img_src})

            img_list_threshold=[]
            img_list_prob = []
            for c in range(self.num_classes):
                img_prob = output[:,c].reshape(img_src.shape[0], img_src.shape[1])
                img_list_threshold.append(img_prob > self.threshold[c])
                img_list_prob.append(img_prob)

            shape = img_src.shape
            img_sum = np.zeros_like(img_src[:,:,0])
            for c in range(self.num_classes):
                img_thresh = img_list_threshold[c]*(c+1)
                img_sum = img_sum + img_thresh

            segmentation = img_sum
            overlay_seg = overlay_segmentation(img_src,segmentation,self.color_seg)
            
            ################################################
            # output configuration
            output = {}
            output['img_avm'] = img_src # 8UC3
            output['img_overlay'] = overlay_seg

            dt = time.time() - start_time

            return output
        else:
            return False


def overlay_segmentation(input_image, segmentation, color_dict):
    width, height = segmentation.shape
    output = scipy.misc.toimage(segmentation)
    output = output.convert('RGBA')
    for x in range(0, width):
        for y in range(0, height):
            if segmentation[x, y] in color_dict:
                output.putpixel((y, x), color_dict[segmentation[x, y]])
            elif 'default' in color_dict:
                output.putpixel((y, x), color_dict['default'])

    background = scipy.misc.toimage(input_image)
    background.paste(output, box=None, mask=output)

    return np.array(background)