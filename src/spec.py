def spec(x, y, color="run ID"):
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"name": "data"},
        "layer": [
            {
                "mark": "line",
                "params": [
                    {
                        "bind": "legend",
                        "name": "legend_selection",
                        "select": {
                            "on": "mouseover",
                            "type": "point",
                            "fields": [color],
                        },
                    },
                    {
                        "bind": "legend",
                        "name": "hover",
                        "select": {
                            "on": "mouseover",
                            "type": "point",
                            "fields": [color],
                        },
                    },
                    {"bind": "scales", "name": "grid", "select": "interval"},
                ],
            }
        ],
        "width": 600,
        "height": 400,
        "encoding": {
            "x": {"type": "quantitative", "field": x},
            "y": {"type": "quantitative", "field": y},
            "color": {"type": "nominal", "field": color},
            "opacity": {
                "value": 0.1,
                "condition": {
                    "test": {
                        "and": [{"param": "legend_selection"}, {"param": "hover"}]
                    },
                    "value": 1,
                },
            },
        },
    }
