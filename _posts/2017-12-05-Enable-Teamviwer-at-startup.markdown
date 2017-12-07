---
layout: post
title:  "Enalbe Teamviwer at Starup"
date:   2017-12-05 22:10 + 0100
categories: Linux Serive
---


In the file `/etc/systemd/system/teamviewerd.service`, change `Restart = on-abort` to `Restart = always`

```
[Unit]
Description = TeamViewer remote control daemon
After = NetworkManager-wait-online.service network.target network-online.target dbus.service
Wants = NetworkManager-wait-online.service network-online.target
Requires = dbus.service

[Service]
Type = forking
PIDFile = /var/run/teamviewerd.pid
ExecStart = /opt/teamviewer/tv_bin/teamviewerd -d
# Changed by the user
#Restart = on-abort#the original
# changed to
Restart = always
# end of change
StartLimitInterval = 60
StartLimitBurst = 10

[Install]
WantedBy = multi-user.target
```

Reload the systemd daemon, followed by a restart of the service:

```
sudo systemctl daemon-reload
sudo systemctl restart service.service
```

Reference:
===========

[How To Configure a Linux Service to Start Automatically After a Crash or Reboot – Part 1: Practical Examples](https://www.digitalocean.com/community/tutorials/how-to-configure-a-linux-service-to-start-automatically-after-a-crash-or-reboot-part-1-practical-examples)