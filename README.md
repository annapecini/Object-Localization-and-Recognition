# Object-Localization-and-Recognition

The goal of this project is to develop a method for image classiffcation and object localization.The method will use a framework that is similar to the R-CNN (region-based convolutional neural network) model proposed by Girshick et al. as discussed to the following papers:
* [R. Girshick, J. Donahue, T. Darrell, J. Malik, \Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation," IEEE Conference on Computer Vision and Pattern Recognition, 580-587, June 23-28, 2014.](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf)
* [R. Girshick, J. Donahue, T. Darrell, J. Malik, \Region-Based Convolutional Networks for Accurate Object Detection and Segmentation," IEEE Transactions on Pattern Analysis and Machine Intelligence, 38(1):142-158, January 2016.](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7112511) <br/><br/>
We have used a ResNet-50 network that was pre-trained on the [ImageNet](http://www.image-net.org) data set to extract visual features. We have trained a 2-layer feed-forward neural network classifier. Selective Search region proposal algorithm is used to predict the bounding boxes of targets in test images. Quantitative performance evaluation is done in two stages: 
* Computing the confusion matrix, precision and recall for each object type, and the overall accuracy in terms of the percentage of correctly classified test images
* Computing the percentage of correctly classified and localized test images. 

 <h2 class="display-5">Contributors:</h2>
  <p>
    <ul>
        <li>
            <a href="https://github.com/annapecini" title="annapecini">Ana Pecini</a>
        </li>
        <li>
            <a href="https://github.com/endimerkuri" title="EndiMerkuri">Endi Merkuri</a>
        </li>
        <li>
            <a href="https://github.com/atakann" title="AtakanSerbes">Atakan Serbes</a>
        </li>
    </ul>
  </p>
