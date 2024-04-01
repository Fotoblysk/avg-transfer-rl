#!/usr/bin/env python

import runner_src.core as app
import config

app_c = app.Core(config.configs)
app_c.menu()