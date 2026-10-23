# astrbot_plugin_saucefinder

AstrBot 反向图片搜索插件：发送图片，自动通过 Yandex Images 反向搜索，返回图片出处（原图链接、画师/Coser 主页署名）。

## 用法

- `sauce` + 图片（同一条消息带图）：反搜该图片
- 回复一张图片并输入 `sauce`：反搜被回复的图片

## 安装

将本仓库克隆到 AstrBot 的 `data/plugins/` 目录下，或通过 AstrBot 插件市场安装。

依赖：见 `requirements.txt`（需要 `httpx`）。

## 说明

- 仅使用 Yandex 公开网页结果，不做任何账号建库、人脸库或跨平台身份聚合。
- 结果来源于公开网页，由 Yandex 相似图索引返回，命中与否取决于图片是否曾被公开发布。
- Yandex 无官方 API，网页结构变更可能导致解析失效；遇到问题时请提 issue。
