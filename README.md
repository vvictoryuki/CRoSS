# CRoSS

The official implementation of the paper:

"**CRoSS**: Diffusion Model Makes **C**ontrollable, **Ro**bust and **S**ecure Image **S**teganography"

By [Jiwen Yu](https://vvictoryuki.github.io/website/), [Xuanyu Zhang](https://villa.jianzhang.tech/people/xuanyu-zhang-%E5%BC%A0%E8%BD%A9%E5%AE%87/), [Youmin Xu](https://zirconium2159.github.io/), [Jian Zhang](https://jianzhang.tech/).

![](./method-cross-v2_00.png)

Inspired by recent developments in diffusion models, we propose a novel image steganography framework named **C**ontrollable, **Ro**bust, and **S**ecure Image **S**teganography (**CRoSS**). This framework offers significant advantages in **controllability** over container images, **robustness** against complex degradation during transmission of container images, and enhanced **security** compared to cover-based image steganography methods. Importantly, these benefits are achieved **without requiring additional training**.

The code will be coming soon.

### Results about Robustness 

Visual comparisons of our **CRoSS** and other methods under two real-world degradations, namely `WeChat` and `Shoot`. Obviously, our method can reconstruct the content of secret images, while other methods exhibit significant color distortion or have completely failed. 

*Details and more results can be found in our paper.*

![](./robust_00.png)

### Results about Security 

Following are deep steganalysis results by the latest [SID](http://www.ws.binghamton.edu/fridrich/research/Scale-1.12.16.pdf). As the number of leaked samples increases, methods whose detection accuracy curves grow more slowly and approach $50\%$ exhibit higher security. The right is the recall curve of different methods under the [StegExpose](https://arxiv.org/pdf/1410.6656v1.pdf) detector. The closer the area enclosed by the curve and the coordinate axis is to 0.5, the closer the method is to the ideal evasion of the detector. 

*Details and more results can be found in our paper.*

![](./security.png)

### Citation

coming soon ...
