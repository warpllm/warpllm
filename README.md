# warpllm

A warp-speed, robust AI gateway written for rust, node, and python applications - built for planet scale by the community.

[![Join Discord](https://img.shields.io/badge/Discord-Join%20the%20warpllm%20community-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/tSSQTxFnsC)
[![Reddit](https://img.shields.io/badge/Reddit-r%2Fwarpllm-FF4500?style=for-the-badge&logo=reddit&logoColor=white)](https://www.reddit.com/r/warpllm/)

## Mission

This project is to lay out the most resilient open source productionization layer for AI-deployments. Designed for you if you want:

1.  To work with multiple AI providers or your own models.
1.  To keep your AI services up and running with 0 downtime.
1.  Speed (minimal overhead latency).
1.  A granular view of your metrics (uptime, P95 latency, costs, etc).
1.  Control over:
    1.  Where your data goes.
    1.  Your AI budget across providers.

## Community

> [!IMPORTANT]
> **warpllm is community-led.**
>
> The roadmap, examples, integrations, and rough edges should be shaped in the open by the people building with it. Bring ideas, questions, provider requests, bug reports, benchmarks, and experiments.

### Contributing

I'm setting up this up! We're excited to have you join us in building this out together. In the meantime, there are a couple things you can do:

1.  **Star this repo**: We appreciate visibility on the project.
1.  **Share your thoughts online**: Post in our discord or reddit community! Your opinion can help others, and we're always listening.

## Layers

1.  **An SDK** - provide a request and we translate it to work with different providers and models out of box.
1.  [Coming Soon] **A proxy** - run a self-hosted proxy that allows you to load balance, failover, etc:
    1.  [Coming Soon] **Failover** - define multiple models to handle outages / errors
    1.  [Coming Soon] **Load Balancing** - define a % of requests to be handled per model
    1.  [Coming Soon] **Prompt Response Caching** - define a TTL and avoid paying twice for the same prompt

## Key focus points

1.  **Native SDK support** - Written once in rust, compiled for maximum performance, available for rust/typescript/python.
1.  **Self hostable** - Avoid vendor lock-in (e.g. from cloud provider or model provider), or data leaving your infra.
1.  **Warp-speed execution** - What we named ourselves after. Machine level code, faster than a typescript or python native library.
1.  **Compact file size** - Pre-compiled into binary format, not verbose text files.

## Roadmap

[Coming Soon] This is in our Github issues. Add a comment if you see something missing or want to prioritize something important to you.

## Quickstart

In progress.

## License

The warpllm core is open source under the [Apache License 2.0](https://github.com/warpllm/warpllm/blob/main/LICENSE).
