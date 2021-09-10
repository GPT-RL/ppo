using Pluto
using ArgParse


s = ArgParseSettings()
@add_arg_table! s begin
    "--notebook"
    help = "notebook to open"
    "--secret"
    help = "secret to use for session"
end

args = parse_args(ARGS, s)

notebook = args["notebook"]
if !isnothing(notebook)
    notebook = "/workspace/notebooks/$(notebook)"
end

options = Pluto.Configuration.from_flat_kwargs(
    host = "0.0.0.0",
    port = 7777,
    launch_browser = false,
    notebook = notebook,
)

secret = args["secret"]
if isnothing(secret)
    Pluto.run(options)
else
    sess = Pluto.ServerSession(; secret = secret, options = options)
    Pluto.run(sess)
end
