install docker desktop

to start the docker server:
1. cd to server folder
2. run `docker build -t server .` to build
3. run `docker run -p 5000:5000 server` to run

to send requests:
1. in a different terminal window, cd to raspberryPi\camera_stuff
2. run `curl -X POST http://10.155.30.209:5000/inference \  -F "original=@processed_face.jpg" \  -F "modified=@overlay_output.png"`