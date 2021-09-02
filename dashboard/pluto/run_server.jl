using Pluto

notebook = length(ARGS) > 0 ? "/workspace/notebooks/$(ARGS[1])" : nothing

Pluto.run(;
    host="0.0.0.0",
    port=7777,
    launch_browser=false,
    notebook=notebook
)
