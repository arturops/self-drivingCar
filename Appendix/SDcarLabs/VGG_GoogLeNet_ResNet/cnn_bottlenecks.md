
## Bottlenecks of all networks

Explanation for why to use bottlenecks from Udacity:

_"**Bottleneck Features -** Unless you have a very powerful GPU, running feature extraction on these models will take a significant amount of time. To make things easier we've precomputed bottleneck features for each (network, dataset) pair. This will allow you to experiment with feature extraction even on a modest CPU. You can think of bottleneck features as feature extraction but with caching. Because the base network weights are frozen during feature extraction, the output for an image will always be the same. Thus, once the image has already been passed through the network, we can cache and reuse the output."_

## Downloads of bottlenecks for 100 features each

1. [VGG](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5834b432_vgg-100/vgg-100.zip)
2. [InceptionV3](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5834b498_inception-100/inception-100.zip)
3. [ResNet](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5834b634_resnet-100/resnet-100.zip)