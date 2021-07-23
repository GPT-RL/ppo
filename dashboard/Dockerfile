FROM julia:1.6.1-buster AS main
WORKDIR /workspace
COPY ./pluto/env/* ./
RUN julia --project=. -e "using Pkg; Pkg.instantiate()"


FROM main AS builder
RUN apt-get update && apt-get install -y build-essential
RUN julia --project=. build_sysimage.jl


FROM main as final
COPY --from=builder /workspace/sys_dashboard.so ./
COPY ./pluto/run_server.jl .

EXPOSE 7777
ENTRYPOINT ["julia", "--project=.", "/workspace/run_server.jl"]
