import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as knnClassifier from '@tensorflow-models/knn-classifier';

const App = () => {
  const [identity, setIdentity] = useState(0);
  const [classes, setClasses] = useState([]);
  const [text, setText] = useState('');
  const [uploadedModel, setUploadedModel] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const webcamRef = useRef(null);
  const trainingCardsRef = useRef(null);
  const predictionsRef = useRef(null);
  const confidenceRef = useRef(null);
  const wordInputRef = useRef(null);
  const inputClassNameRef = useRef(null);
  const mobilenetModelRef = useRef(null);
  const knnClassifierModelRef = useRef(null);
  const webcamInputRef = useRef(null);

  const start = async () => {
    const createKNNClassifier = async () => await knnClassifier.create();
    const createMobileNetModel = async () => await mobilenet.load();
    const createWebcamInput = async () => {
      const webcamElement = await webcamRef.current;
      return await tf.data.webcam(webcamElement);
    };

    const mobilenetModel = await createMobileNetModel();
    const knnClassifierModel = await createKNNClassifier();
    const webcamInput = await createWebcamInput();

    mobilenetModelRef.current = mobilenetModel;
    knnClassifierModelRef.current = knnClassifierModel;
    webcamInputRef.current = webcamInput;

    const togglePause = () => {
      setIsPaused((prevState) => !prevState);
    };

    document.getElementById('pause-button').addEventListener('click', togglePause);

    const addClass = () => {
      const className = inputClassNameRef.current.value;
      const found = classes.some((el) => el.name === className);
      if (!found) {
        setIdentity((prevIdentity) => prevIdentity + 1);
        setClasses((prevClasses) => [
          ...prevClasses,
          { id: identity + 1, name: className, count: 0 },
        ]);
      }

      trainingCardsRef.current.innerHTML += `<div><div><h3>ID : <span>${className}</span></h3><h3>Imagenes: <span id="images-${identity + 1}">0</span></h3></div><div><button id="${identity + 1}">Añadir</button></div></div>`;

      window.scrollTo(0, document.body.scrollHeight);

      document
        .getElementById((identity + 1).toString())
        .addEventListener('click', () => addDatasetClass(identity + 1));
      inputClassNameRef.current.value = '';
    };

    const clearWord = () => {
      wordInputRef.current.value = '';
      setText('');
    };

    const imageClassificationWithTransferLearningOnWebcam = async () => {
      while (true) {
        if (!isPaused && knnClassifierModelRef.current.getNumClasses() > 0) {
          const img = await webcamInputRef.current.capture();
          const activation = mobilenetModelRef.current.infer(img, 'conv_preds');
          const result = await knnClassifierModelRef.current.predictClass(activation);

          if (uploadedModel) {
            console.log(result.label);
            predictionsRef.current.innerHTML = result.label;
            confidenceRef.current.innerHTML = Math.floor(result.confidences[result.label] * 100);
            setText((prevText) => prevText + result.label);
          } else {
            try {
              console.log(classes[result.label - 1].name);
              predictionsRef.current.innerHTML = classes[result.label - 1].name;
              confidenceRef.current.innerHTML = Math.floor(result.confidences[result.label] * 100);
              setText((prevText) => prevText + classes[result.label - 1].name);
            } catch (err) {
              console.log(result.label - 1);
              predictionsRef.current.innerHTML = result.label - 1;
              confidenceRef.current.innerHTML = Math.floor(result.confidences[result.label] * 100);
              setText((prevText) => prevText + (result.label - 1).toString());
            }
          }

          wordInputRef.current.value = text;

          img.dispose();
        }
        await tf.nextFrame();
        await sleep(3000);
      }
    };

    await imageClassificationWithTransferLearningOnWebcam();
  };

  const addDatasetClass = async (classId) => {
    const img = await webcamInputRef.current.capture();
    const activation = mobilenetModelRef.current.infer(img, 'conv_preds');
    knnClassifierModelRef.current.addExample(activation, classId);

    const classIndex = classes.findIndex((el) => el.id === classId);
    const currentCount = classes[classIndex].count;
    const updatedClasses = [...classes];
    updatedClasses[classIndex].count = currentCount + 1;
    setClasses(updatedClasses);

    const tempId = `images-${classId}`;
    document.getElementById(tempId).innerHTML = currentCount + 1;

    img.dispose();
  };

  useEffect(() => {
    start();
  }, []);

  const togglePause = () => {
    setIsPaused((prevState) => !prevState);
  };

  const addClass = () => {
    const className = inputClassNameRef.current.value;
    const found = classes.some((el) => el.name === className);
    if (!found) {
      setIdentity((prevIdentity) => prevIdentity + 1);
      setClasses((prevClasses) => [
        ...prevClasses,
        { id: identity + 1, name: className, count: 0 },
      ]);
    }

    trainingCardsRef.current.innerHTML += `<div><div><h3>ID : <span>${className}</span></h3><h3>Imagenes: <span id="images-${identity + 1}">0</span></h3></div><div><button id="${identity + 1}">Añadir</button></div></div>`;

    window.scrollTo(0, document.body.scrollHeight);

    document
      .getElementById((identity + 1).toString())
      .addEventListener('click', () => addDatasetClass(identity + 1));
    inputClassNameRef.current.value = '';
  };

  const clearWord = () => {
    wordInputRef.current.value = '';
    setText('');
  };

  const sleep = (ms) => {
    return new Promise((resolve) => setTimeout(resolve, ms));
  };

  return (
    <div>
      <video autoPlay playsInline muted ref={webcamRef} id="webcam" width="224" height="224" />
      <div ref={trainingCardsRef} />
      <div ref={predictionsRef} />
      <div ref={confidenceRef} />
      <input type="text" id="inputClassName" ref={inputClassNameRef} />
      <button id="add-button" onClick={addClass}>
        Add
      </button>
      <button id="pause-button" onClick={togglePause}>
        Pause
      </button>
      <input type="text" id="word-input" ref={wordInputRef} />
      <button id="clear-button" onClick={clearWord}>
        Clear
      </button>
    </div>
  );
};

export default App;
