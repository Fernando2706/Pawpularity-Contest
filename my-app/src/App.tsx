import React, { useEffect, useRef, useState } from 'react';
import styled from '@emotion/styled'
import axios from 'axios';
import { DotLoader } from 'react-spinners';
function App() {
  const [files,setFiles] = useState<File>()
  const [url,setUrl] = useState<string>("")
  const [loading,setLoading] = useState<boolean>(false)
  const [firstOkay, setFirstOkay] = useState<boolean>(false)
  const [secondOkay, setSecondOkay] = useState<boolean>(false)
  const [message, setMessage] = useState<string>("")
  const [message1, setMessage1] = useState<string>("")
  const [message2, setMessage2] = useState<string>("")

  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(()=>{
    if(files){
      setUrl(URL.createObjectURL(files))
    }
  },[files])  

  useEffect(()=>{
    if(secondOkay && files){
      setLoading(true)
      setTimeout(()=>{
        const formData = new FormData()
        formData.append("file",files)
        axios.post("http://localhost:5000/getPawpularity",formData,{
          headers:{
          'Content-Type': 'multipart/form-data'
          },
        }).then(res=>{
          console.log(res);
          setMessage(res.data.predicts)
          setMessage1(res.data.predict1)
          setLoading(false)
        }).catch(err=>{
          console.error(err);
          
        })
      },3000)
    }
  },[secondOkay])

  if(loading){
    return(
      <MainContainer>
        <DotLoader/>
      </MainContainer>
    )
  }

  return (
    <MainContainer>
      {(!firstOkay&&!secondOkay) && (
        <>
          <Title>¡Hacer una predicción!</Title>
          <Button type="button" onClick={(e)=>{
            e.preventDefault()
            setFirstOkay(true)
          }}>¡Empecemos!</Button>
        </>
      )}

      {(firstOkay && !secondOkay )&& (
        <>
        <Title>Primero selecciona la imagen sobre la que quieres hacer la predicción</Title>
        <Input 
          type='file'
          ref={inputRef}
          onChange={(e)=>{
            const file = e.target.files
            if(file){
              setFiles(file[0])
            }
          }}
        />
        {url !== "" && (
          <>
            <Image src={url} />
            <span>¿Quieres usar esta foto?</span>
            <CheckBox onClick={(e)=>setSecondOkay(!secondOkay)} isActive={secondOkay}/>
          </>
        )}

        </>
      )}
      {secondOkay && (
        <>
          <Title>¡Se ha completado la predicción!</Title>
          <span>La popuralidad de la foto según los metadatos es: {message}</span>
          <span>La popuralidad de la foto según una CNN es: {message1}</span>
          <Button type='button' onClick={(e)=>{
            e.preventDefault()
            setSecondOkay(false)
            setFirstOkay(true)
            setFiles(undefined)
            setLoading(false)
            setUrl("")
          }}>¿Quieres volver a intentarlo?</Button>
        </>
      )}
      </MainContainer>
  );
}

export default App;


const MainContainer = styled.div`
  width:100%;
  min-height:100vh;
  margin: 0 auto;
  display:flex;
  flex-direction:column;
  align-items:center;
  justify-content:center;
  background-color:#CACFD2;
  transition: all 1s ease;
  `
const Input = styled.input`
  width:fit-content;
  height:fit-content;
  border-radius:5px;
  padding:10px;
  border: 1px solid #ccc; 
  margin:10px 0;
`
const Image = styled.img`
  margin:10px;
  max-width:300px;
  max-height:300px;
  object-fit:contain;
`

const Title = styled.h1`
font-size:24px;
text-align:center;
color:#273746;
`

const Button = styled.button`
  margin: 10px auto;
  width:fit-content;
  padding:5px 10px;
  font-size:15px;
  border-radius:6px;
  background-color: #2E4053;
  color:#ABEBC6;
  cursor:pointer;
`

const CheckBox = styled.div<{
  isActive:boolean;
}>`
  width:20px;
  height:20px;
  border-radius:3px;
  border: 1px solid #273746;
  cursor:pointer;
  margin:15px;
  position: relative;
  &::after {
    content:${props=>props.isActive ? `"✓"` : `"X"` };
    color: ${props=>props.isActive ? "green" : "red"};
    position:absolute;
    top:50%;
    left:50%;
    transform:translate(-50%,-50%);
  }
`