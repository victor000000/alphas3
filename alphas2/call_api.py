
    def call_doubao_api(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Call Doubao API to generate templates"""
        headers = {
            'Authorization': 'Bearer 573299a6-28a8-47e6-a9ff-4130d7b9ead3',
            'Content-Type': 'application/json'   
        }
        model_str_list = [
            "deepseek-v3-1-250821",
            "doubao-seed-1-6-250615",
            "doubao-seed-1-6-flash-250828",
            "doubao-seed-1-6-thinking-250715",
            "deepseek-r1-250528",
            "kimi-k2-250905"
        ]
        model_str = random.choice(model_str_list)
        logger.info(f"Selected model for Doubao API: {model_str}")
        payload = {
            "model": model_str, 
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in quantitative finance and WorldQuant Brain alpha expressions. Generate valid, creative alpha expression templates with proper syntax."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        }
        for attempt in range(max_retries):
            try:
                logger.info(f"Doubao API call attempt {attempt + 1}/{max_retries}")
                response = requests.post(
                    'https://ark.cn-beijing.volces.com/api/v3/chat/completions',
                    headers=headers,
                    json=payload,
                    timeout=80
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    logger.info("Doubao API call successful")
                    return model_str, content
                else:
                    logger.warning(f"Doubao API error: {response.status_code} - {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt + 30)
                        continue
                    return model_str, None
                    
            except Exception as e:
                logger.error(f"Doubao API call failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return model_str, None
        return model_str, None

    def call_hunyuan_api(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        # -*- coding: utf-8 -*-
        """Call Hunyuan API to generate templates"""
        import os
        import json
        import types
        from tencentcloud.common import credential
        from tencentcloud.common.profile.client_profile import ClientProfile
        from tencentcloud.common.profile.http_profile import HttpProfile
        from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
        from tencentcloud.hunyuan.v20230901 import hunyuan_client, models
        try:
            # 密钥信息从环境变量读取，需要提前在环境变量中设置 TENCENTCLOUD_SECRET_ID 和 TENCENTCLOUD_SECRET_KEY
            # 使用环境变量方式可以避免密钥硬编码在代码中，提高安全性
            # 生产环境建议使用更安全的密钥管理方案，如密钥管理系统(KMS)、容器密钥注入等
            # 请参见：https://cloud.tencent.com/document/product/1278/85305
            # 密钥可前往官网控制台 https://console.cloud.tencent.com/cam/capi 进行获取
            cred = credential.Credential("AKIDfw2wdNzvgHj2A91nRu7E9rbHoANp7GDz", "iFev67wa2dBg8TguxufOwISghW9apsjN")
            # 使用临时密钥示例
            # cred = credential.Credential("SecretId", "SecretKey", "Token")
            # 实例化一个http选项，可选的，没有特殊需求可以跳过
            httpProfile = HttpProfile(reqTimeout=80)
            httpProfile.endpoint = "hunyuan.tencentcloudapi.com"

            # 实例化一个client选项，可选的，没有特殊需求可以跳过
            clientProfile = ClientProfile()
            clientProfile.httpProfile = httpProfile
            # 实例化要请求产品的client对象,clientProfile是可选的
            client = hunyuan_client.HunyuanClient(cred, "", clientProfile)
            

            # 实例化一个请求对象,每个接口都会对应一个request对象
            req = models.ChatCompletionsRequest()
            model_str_list = [
                "hunyuan-t1-latest",
                "hunyuan-turbos-latest"
            ]
            model_str = random.choice(model_str_list)
            logger.info(f"Selected model for Hunyuan API: {model_str}")
            params = {
                "Model": "hunyuan-t1-latest",
                "Messages": [
                    {
                        "Role": "system",
                        "Content": "You are an expert in quantitative finance and WorldQuant Brain alpha expressions. Generate valid, creative alpha expression templates with proper syntax."
                    },
                    {
                        "Role": "user",
                        "Content": prompt
                    }
                ]
            }
            req.from_json_string(json.dumps(params))

            # 返回的resp是一个ChatCompletionsResponse的实例，与请求对象对应
            # resp = client.ChatCompletions(req)
            # # 输出json格式的字符串回包
            # if isinstance(resp, types.GeneratorType):  # 流式响应
            #     for event in resp:
            #         print(event)
            # else:  # 非流式响应
            #     print(resp)
        except TencentCloudSDKException as err:
            print(err)
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Hunyuan API call attempt {attempt + 1}/{max_retries}")

                try:
                    resp = client.ChatCompletions(req)
                except TencentCloudSDKException as err:
                    print(err)
                    logger.error(f"Hunyuan API call failed: {err}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt + 30)
                        continue
                    return model_str, None

                content = resp.Choices[0].Message.Content
                logger.info("Hunyuan API call successful")

                return model_str, content


                    
            except Exception as e:
                logger.error(f"Hunyuan API call failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return model_str, None
        return model_str, None

    def call_qianwen_api(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Call Qianwen API to generate templates"""
        headers = {
            'Authorization': 'Bearer sk-7e9f1b1094ea48a9aeb4b39320adf789',
            'Content-Type': 'application/json'
        }
        model_str_list = [
            "qwen-max-latest",
            "qwen-max",
            "qwen-plus-latest",
            "qwen-plus",
        ]
        model_str = random.choice(model_str_list)
        logger.info(f"Selected model for Qianwen API: {model_str}")
        payload = {
            "model": model_str,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in quantitative finance and WorldQuant Brain alpha expressions. Generate valid, creative alpha expression templates with proper syntax."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        }
        for attempt in range(max_retries):
            try:
                logger.info(f"Qianwen API call attempt {attempt + 1}/{max_retries}")
                response = requests.post(
                    'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
                    headers=headers,
                    json=payload,
                    timeout=80
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    logger.info("Qianwen API call successful")
                    return model_str, content
                else:
                    logger.warning(f"Qianwen API error: {response.status_code} - {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt + 30)
                        continue
                    return model_str, None
                    
            except Exception as e:
                logger.error(f"Qianwen API call failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return model_str, None
        return model_str, None


###############################################################



        api_callers = [
            self.call_doubao_api,
            self.call_qianwen_api,
            self.call_qianwen_api,
            self.call_hunyuan_api
        ]
        api_caller = random.choice(api_callers)

        # Call xxx API
        ai_model_str, response = api_caller(prompt)
