# astrbot_plugin_bitchfinder

AstrBot 反向图片搜索插件：发送图片，自动通过多引擎反向搜索，返回图片出处与作者账号（含 X 原帖、Coser 主页署名）。

## 用法

`找鸡` + 图片/引用图片

别名：`kfc`

## 示例

发送一张 Coser 图片（如 `sample.png`）并输入 `找鸡`，插件会返回该图的可能出处与作者账号，例如：

```
可能的作者账号（正文署名优先，相似度仅在同图时作数）：
✅ @mamamia1008  [X/Twitter] [正文署名] 图像相似度 98%
      页面: https://x.com/mamamia1008/status/...
      检索: https://x.com/search?q=from%3Amamamia1008
```

![示例](sample.png)
![证据](evidence.png)


## 说明

- 仅使用 Yandex 公开网页结果，不做任何账号建库、人脸库或跨平台身份聚合。
- 结果来源于公开网页，由 Yandex 相似图索引返回，命中与否取决于图片是否曾被公开发布。
- Yandex 无官方 API，网页结构变更可能导致解析失效；遇到问题时请提 issue。
