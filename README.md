# TrashDetection-HuaweiCloud
“华为云杯”2020深圳开放数据应用创新大赛·生活垃圾图片分类 Pytorch版本基础配置文件

# How to use
1. 已经编写好customize_service.py的pytorch基础引入方式，需要修改以适配你自己的网络输出
```
  1. def YourNet()
    从 YourModelDict/model.py 导入你自己的model
    
  2. Class PTVisionService(PTServingBaseService):
  
        def _preprocess(self, data):
            我自己进行数据预处理Normalizer和Resizer的数据格式为: sample={'img': img, 'img_name': img_name}
            修改为你自己进行归一化和Resizer的数据格式

        def _inference(self, data):
            修改网络输出部分:
                if torch.cuda.is_available():
                    scores, classification, predict_bboxes = self.model(padded_imgs.cuda().float())
                else:
                    scores, classification, predict_bboxes = self.model(padded_imgs.float())
            为自己的网络输出

        def _postprocess(self, data):
            修改网络输出格式，以匹配‘save_json’函数所需输入，形成结果

        def inference(self, data):
            无需更改
      
 ```
 2. 如何本地测试
 ```
  1. 修改 customizer_service.py ---> 
      Class PTVisionService(PTServingBaseService): 为  Class PTVisionService(object):
  
  2. 删除 customizer_service.py ---> 
      import部分modelarts相关的依赖
  
  3. 添加如下代码在customize_service.py的底部:
      if __name__ == '__main__':
      m = PTVisionService('model_name', 'your/model/path/model.pth')
      data={'images':{'0':'./train_val/VOC2007/JPEGImages/img_12726.jpg'}}
      m.inference(data)
 ```

# TIPS
  1. 提交报错 build model image failed，是因为模型编译失败 ----> `config.json`中指定的‘dependencies’无效<br>
  ModelArts内置torch==1.0.0, torchvision < 0.3.0，注意适配
