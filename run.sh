#! /bin/bash

export PYTHONPATH=app
exec bokeh serve --show app --args "$@"
