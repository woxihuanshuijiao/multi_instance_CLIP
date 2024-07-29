# High Resolution Multimodal Model for Pathology

`This README(if this page still available) is only for your reference.`



## What is it?

A multimodal model that can handle max 1920*1080 pixel pathological images, based on multi-instance learning.

- backbone: ViT-L-14-336
- task: captioning or more...

check the illustration:

![work diagram](https://picfiles.alphacoders.com/650/650380.png "work diagram")



## Why did this?

The pixel of pathological images is so massive and every pixel contains lot of details. Nowadays generally process the input only by resizing to a fixed [height, width], it causes the tissue out of shape and confuses the model. So our work aims to let the model learn more effectively.

![illustration](https://picfiles.alphacoders.com/650/650381.png "illustration")


## What's the next? (on going)

- organize more and more data. 
- explore a better way to supervise the bag encoder layer.
- figure out the possibility on zero-shot or few-shot.