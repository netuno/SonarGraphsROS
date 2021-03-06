#!/usr/bin/env python
import rospy
from extractor import GaussianFeatureExtractor
from sensor_msgs.msg import Image
import time

class SonarNodes(object):
    
    #Initializes the node
    def __init__(self):
    
        #Parameters:
        self.auto_thresholding = True
        self.save_gaussian_image = False
        self.gaussian_image_name = ''
        self.merge = False
        self.write_graph = False
        self.export_graph_name = ''
    
        rospy.init_node('sonar_node', anonymous=True)
        self.extract = GaussianFeatureExtractor()
    
    #Starts to listen to sonar images
    def sonar_listener(self):
        rospy.Subscriber("sonar_image_gray", Image, self.callback)

    def gaussian_publisher(self):
        self.pub_gauss = rospy.Publisher('gaussian_image', Image, queue_size=10)
        
    def graph_publisher(self):
        self.pub_graph = rospy.Publisher('graph_image', Image, queue_size=10)
        
    def callback(self, data):
        #Initiliazes time
        init = time.time()
        
        #img, new_img = self.extract.initImage(data)
        self.extract.initImage(data, self.auto_thresholding)
        
        #publishing gaussian image
        gaussian_image = self.extract.createSegments()
        #gaussian_image = self.extract.createSegmentsFloodFill()
        self.pub_gauss.publish(self.extract.convertImage(gaussian_image))
        
        #publishing graph image
        graph_image = self.extract.drawGraph(self.merge, self.write_graph, self.export_graph_name)
        self.pub_graph.publish(self.extract.bridge.cv2_to_imgmsg(graph_image, "rgb8"))
        
        
        print time.time() - init
        

if __name__ == '__main__':
    try:
        sonar_node = SonarNodes()
        sonar_node.gaussian_publisher()
        sonar_node.graph_publisher()
        sonar_node.sonar_listener()
        rospy.spin()
        
    except rospy.ROSInterruptException:
        pass

