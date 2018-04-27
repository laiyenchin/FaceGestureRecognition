Face Gesture Recognition using Python, Opencv, Fisherface
1.Collecting the data for training
  Execute “collectdata.py” with the command
  python collectdata.py —-shape-predictor shape_predictor_68_face_landmarks.dat
[in collectdata.py]
  need to change “facefilename” properly, the number after dot is an ID
  IDs represents class labels
  press key “c” to save the image to folder “data”

2.Training with the collected data
  Execute “trainer.py” and will get a “.yml” trained data

3.Recognition
  Execute “FaceGestureRecognition.py” with the command
  python FaceGestureRecognition.py —-shape-predictor shape_predictor_68_face_landmarks.dat
*sudo rm -rf .DS_Store if the .DS_Store file appear