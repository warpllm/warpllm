#[path = "providers/openai/common.rs"]
mod openai_common;

#[path = "providers/openai/chat_completions/mod.rs"]
mod openai_chat_completions;

#[path = "providers/openai/config_env.rs"]
mod openai_config_env;

#[path = "providers/deepseek/chat_completions/mod.rs"]
mod deepseek_chat_completions;

#[path = "providers/deepseek/config_env.rs"]
mod deepseek_config_env;

#[path = "providers/opencode/config_env.rs"]
mod opencode_config_env;

#[path = "providers/openrouter/config_env.rs"]
mod openrouter_config_env;

#[path = "providers/openrouter/chat_completions/mod.rs"]
mod openrouter_chat_completions;
