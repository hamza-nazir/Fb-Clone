import * as React from 'react';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import IconButton from '@mui/material/IconButton';
import CloseIcon from '@mui/icons-material/Close';
import { useRef } from 'react';
import Tooltip from '@mui/material/Tooltip';
import LabelIcon from '@mui/icons-material/Label';
import CircularWithValueLabel from "../Components/ProgressBar";
import { useState } from 'react';
import { useContext } from 'react';
import { Context } from '../Hook/Context';
import useAuth from '../Hook/useAuth';
import api from '../Hook/Axios'
const PostPopup = () => {
    const captionText=useRef();
    const currentUser=useAuth()
    const {postPopUp,setPostPopUp,percentage,setPercentage}=useContext(Context)
const [imageURL,setImageURL]=useState("");

const [caption,setCaption]=useState("");
const [imageFile,setImageFile]=useState("");

    function handleImageObject(e){
        setImageFile(e.target.files[0])
       const TempImageURL= URL.createObjectURL(e.target.files[0]);
       setImageURL(TempImageURL);
    }
    const submitFun=()=>{
        console.log(imageFile);
        console.log(caption);
        if(imageFile){
              const formData=new FormData;
        formData.append("caption",caption)
        formData.append("imageFile",imageFile);
         api.post('/upload-data',formData, {
        onUploadProgress: (progressEvent) => {
        const percentCompleted = Math.round( (progressEvent.loaded * 100) / progressEvent.total );
        setPercentage(percentCompleted);
        }})

.then((res) => {
  console.log(res.data);
})
        }else{
             api.post('/upload-caption',{caption})
             .then((res)=>{
                console.log(res.data);
             })
        }
      

    }
   async function captionReq(){
    
        const formData = new FormData();
  formData.append("image", imageFile); 
  const res = await fetch("http://127.0.0.1:5000/caption", {
    method: "POST",
    body: formData,
  });
  const data = await res.json();
  console.log( data.caption);
 setCaption(data.caption);
  


  

}

    return (
       <Dialog open={postPopUp}  maxWidth="sm" fullWidth>
      <DialogTitle>
        <div>
        <h2 className='text-center'>Create Post</h2>
            <CircularWithValueLabel value={percentage} /> 
         </div>
        <IconButton onClick={()=>setPostPopUp(false)} sx={{position: 'absolute',right: 8,top: 20}}>
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      <DialogContent dividers>
        <div className="row">
           
            <div className="col-1">

                {
                    currentUser?.profileImg?(
                <img  className='me-3' src={currentUser?.profileImg} style={{borderRadius:'50%',border:'1px solid black',height:'40px'}} width={40} />

                    ):(
                <img  className='me-3' src="profile.png" style={{borderRadius:'50%',border:'1px solid black'}} width={40} />

                    )
                }
           
           
            </div>
            <div className="col-4 mt-2">
                <b className='ms-2'>{currentUser?.name}</b>
            </div>
        </div>
        <div className='mt-4'>
            <textarea value={caption} onChange={(e)=>setCaption(e.target.value)} id='feed-input' rows={3}  type="text" style={{ resize:'none',border:'none',width:'100%',paddingBottom:'10px'}}  placeholder={`Whats on your mind, ${currentUser?.name}?`} />
        </div>
        <div className='mb-4' style={{display:'flex',justifyContent:'center',alignItems:'center'}} >
            {
            imageURL&&(
            <>
            <img className='rounded-2' src={imageURL} width={400} height={300} />

            </>
                )
            }
        </div>
        {
                         
            imageURL&&(<span id='image-cross-mark' ><i onClick={()=>{setImageURL(null);setImageFile(null)}} className="fa-solid fa-circle-xmark fs-5"></i></span>)
        }

        <div className='row border p-2 rounded-3'>
            <div className="col">Add to your post</div>
            <div className="col-1">
              <input type="file" id='image' onChange={handleImageObject} style={{display:'none'}}/>
              <label htmlFor="image"><i style={{color:'#41B35D',cursor:'pointer'}} className="fa-solid fa-image"></i></label>
            </div>
           
            <div className="col-1">
                {imageFile&&(

                <Tooltip title="Get Caption">
    <LabelIcon 
        onClick={captionReq} 
        style={{ color: 'red', cursor: 'pointer' }} 
    />
</Tooltip>
                )}
            </div>

             <div className="col-1">
                <i className="fa-solid fa-ellipsis"></i>
            </div>
        </div>
        <div className="row">
            <button disabled={!imageURL&&!caption} onClick={submitFun} className='btn btn-primary mt-2'>Post</button>
        </div>
      </DialogContent>
    </Dialog>
  );
}


export default PostPopup
