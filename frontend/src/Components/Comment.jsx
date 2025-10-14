import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import { useContext, useState } from 'react';
import { Context } from '../Hook/Context';
import api from '../Hook/Axios'
import IconButton from '@mui/material/IconButton';
import CloseIcon from '@mui/icons-material/Close';
import { useEffect } from 'react';

const Comment = () => {
           const{selectedPost,comment,setComment,commentsData,setCommentsData,}=useContext(Context);
           const [cmnt,setCmnt]=useState("");
  
function submitComment(post){
    api.post('/comment',{post,cmnt})
    .then((res)=>{
        // setCmnt("")
        setCommentsData(res.data.cmntData);
    })
}

useEffect(() => {
  if (selectedPost?._id) {
    api.post('/get-comments', { postID: selectedPost._id })
      .then(res => {
        setCommentsData(res.data.comments);
        //   setSelectedPost(prev => ({ ...prev, commentsCount: res.data.comments.length }));
      })
     
  }
}, [selectedPost]);

    return (
   
       <Dialog  maxWidth="sm" fullWidth open={comment}>
      <DialogTitle>
        <div>
        <h2 className='text-center'>Create Post</h2>
         </div>
        <IconButton onClick={()=>setComment(false)} sx={{position: 'absolute',right: 8,top: 20}}>
          <CloseIcon />
        </IconButton>
      </DialogTitle>
      <DialogContent dividers>
        
<div className="container" style={{height:'300px',overflowY:'scroll'}}>
<div className="row">
    <div className="col-1 pt-1">
        {
            selectedPost?.owner?.profileImg?(
        <img src={selectedPost?.owner?.profileImg} width={40} style={{height:'40px',borderRadius:'50%'}} />

            ):(
        <img src='/profile.png' width={40} style={{height:'40px',borderRadius:'50%'}}/>

            )
        }
    </div>
    <div className="col">
        <b >{selectedPost?.owner?.name }</b>
   <div onClick={()=>console.log(commentsData)} >{new Date(selectedPost?.createdAt).toLocaleString()}</div>
    </div>
</div>

<div className="row ms-2">
    {selectedPost?.caption}
</div>
{
    selectedPost?.post&&(
        <div className="row">
    <img  src= {selectedPost?.post} alt="" />
</div>
    )
}

<div>
    {
        commentsData.map((comment,index) =>(
        
            <div key={index} className="row border p-1 mt-1 mb-1 rounded">
                <div className="col-2 mt-2" >
                    {comment?.owner?.profileImg?
                    ( <img className='ms-3 mt-1' src={comment.owner.profileImg} width={40} style={{height:'40px',borderRadius:'50%'}}/>)
                    :(<img className='ms-3 mt-1' src='/profile.png' width={40} style={{height:'40px',borderRadius:'50%'}} /> ) }
                </div>
                
                <div  className="col ">
                    <div className="row">
                        <div className="col-12"><b>{comment.owner.name}</b></div>
                         <div className="col text-muted " style={{fontSize:'12px'}}> {new Date(comment.createdAt).toLocaleString()}</div>

                    </div>
                </div>
                

                      <div className='row'>
                        <div className="col-2"></div>
                        <div className="col">
                              {comment.comment}
                            
                        </div>
                

            </div>
            </div>
          
       
        ))
    }

</div>

</div>
<div>
   <textarea onChange={(e)=>setCmnt(e.target.value)} className='form-control mt-2' placeholder='Write a comment...' name="" id="" style={{width:'100%',resize:'none'}}/>
<div style={{display:'flex',justifyContent:'flex-end',position:'relative',bottom:'42px'}}>  <button onClick={()=>submitComment(selectedPost)} disabled={!cmnt} style={{border:'none',backgroundColor:'transparent'}}><i style={!cmnt?{cursor:'pointer'}:{color:'#0866FF',cursor:'pointer'}}  className="fa-brands fa-telegram pe-4 fs-3"></i></button> </div>
</div>



      </DialogContent>
    </Dialog>

    )
}

export default Comment
