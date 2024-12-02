import torch

# 두 .pt 파일을 불러옵니다.
weight1 = torch.load('/workspace/mnt/sda/changhyun/Controlnet_custom/result/sd_image_adapter_4_768/image_adapter/image_adapter_1.pt')
weight2 = torch.load('/workspace/mnt/sda/changhyun/Controlnet_custom/result/sd_image_adapter_4_768/image_adapter/image_adapter_2.pt')

# 구조와 값들이 같은지 다른지 확인합니다.
if weight1.keys() == weight2.keys():
    print("두 weight의 구조는 같습니다.")
else:
    print("두 weight의 구조는 다릅니다.")

for key in weight1.keys():
    if torch.equal(weight1[key], weight2[key]):
        print(f"{key}의 값은 같습니다.")
    else:
        print(f"{key}의 값은 다릅니다.")

