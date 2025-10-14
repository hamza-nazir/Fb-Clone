import { useEffect, useState } from 'react';
import Navbar from '../Components/Navbar'
import { Context } from '../Hook/Context';
import { useLongPress } from "use-long-press";
import { useContext } from 'react';
import Comment from '../Components/Comment';
import api from '../Hook/Axios'
import Friends from '../Components/Friends'
import useAuth from '../Hook/useAuth';
import { Link } from 'react-router-dom';
import DrawerComp from '../Components/DrawerComp';
import EditProfile from '../Components/EditProfile';
import { useParams } from 'react-router-dom';
import LogoutPop from '../Components/LogoutPop';
const Profile = () => {

    const {setComment,userRelation,setUserRelation,userPosts,setUserPosts,userDetails,setUserDetails,setEdit,profileOption,setProfileOption}=useContext(Context);
    

  const currentUser=  useAuth();

    const {id}=useParams();
    useEffect(()=>{
    api.post('/profile-view',{id})
    .then((res)=>{
        if(res.data.sendRequestDocument!=null){
          setUserRelation(res.data.sendRequestDocument);
        }else{
             setUserRelation(null);
        }
    setUserDetails(res.data.user);
    setUserPosts(res.data.userPosts);
// console.log("JEO");
    })
},[id])



function likeFun(postID){
api.post('/like',{postID})
    .then((res)=>{
        const updatedPost = res.data.postToLike;
        setUserPosts((prevPosts) =>
        prevPosts.map((post) => {
        return post._id === updatedPost._id ? updatedPost : post;
        }));
})}


function dislikeFun(postID){
  api.post('/dislike',{postID})
  .then((res)=>{
       const updatedPost =res.data.postTodislike;
        setUserPosts((prevPosts) =>
        prevPosts.map((post) =>
          post._id === updatedPost._id ? updatedPost : post
        ))})
}

function sendRequest(userToSendRequestID){

      api.post('/send-request',{userToSendRequestID})
      .then((res)=>{
        console.log(res.data);
        setUserRelation(res.data.sendRequestDocument);
      })
      
}


function cancelRequest(userToCancelRequestID){

      api.post('/cancel-request',{userToCancelRequestID})
      .then((res)=>{
       setUserRelation(res.data.cancelRequestDocument);
      })
      
      
}

function editProfile(userID){
    if(userID==currentUser?._id){
        setEdit(true)
    }

}


const imagePreview = useLongPress(() => {
   console.log("OK");
  }, {
    threshold: 500, 
  });

    return (
        <div>
             <Comment/>
            <Navbar  />
             <DrawerComp/>
             <LogoutPop/>
             <EditProfile/>
      
        <div onClick={()=>console.log(userRelation)} id='maincoverImage'  className='container-fluid' style={{marginTop:'51px',textAlign:'center', background: "linear-gradient(to bottom, #bdb9bdff, #FFFFFF)"}}>
        {
        userDetails?.coverImg!=null?(
            <img src={userDetails?.coverImg} id='coverImage' className='rounded-2 img-fluid' style={{height:'350px',width:'70%'}} alt="cover" />

        ):(
            <img src="/cover.png" id='coverImage' className='rounded-2 img-fluid' style={{height:'350px',width:'70%'}} alt="cover" />

        )
        }

        </div>

        <div className="row mx-0">
            <div className="col-lg-2 d-none d-lg-block"></div>
    <div className="col-lg-1 col-sm-2"></div>
            <div className="col-md-2 col-3">

                {
                    userDetails?.profileImg!=null?(
                         <img src={userDetails.profileImg} {...imagePreview()} className='img-fluid' style={{
                   cursor:'pointer',
                    width:'180px' , 
                    height:'180px',
                    border:'1px solid black',
                    objectFit:'cover',
                    // objectFit:'contain',
                    borderRadius:'50%',
                    position:'relative',
                    bottom:'40px',
                    // width: '100%'
                }} alt="profile" />
                    ):(
                        <img src="/profile.png" className='img-fluid' style={{
                    maxWidth: 180, 
                    border:'1px solid black',
                    borderRadius:'50%',
                    position:'relative',
                    bottom:'40px',
                    width: '100%'
                }} alt="profile" />
                    )

                }
                
            </div>
            <div className="col-md-2 col-sm-3 col-5 mt-5">
                <b className=''>{userDetails?.name}</b>
                <p  className='small'>{userRelation?.friends?.length||0} friends</p>
            </div>
            <div className="col d-none d-md-block"></div>

{/* <button onClick={()=>console.log(userDetails)}> click</button> */}
            <div className="col-md-2 col-4 mt-5">
     
                   {userDetails?._id===currentUser?._id&&(<button className='btn btn-primary' onClick={()=>{editProfile(userDetails?._id)}} >Edit profile</button>)}

                <div className="row">



               
                {
                userRelation?.friends?.some(user=>user._id===currentUser?._id)?(
                    <>
                   <p className='border rounded-2 bg-primary text-center' style={{width:'80px',padding:'5px',color:'white'}}>friends</p>
                    </>
                ):userDetails?._id!==currentUser?._id?(
                     
                    userRelation?.requests?.some(user=>user._id===currentUser?._id)?(
                    <div className="col-auto">
                    <button onClick={()=>cancelRequest(userDetails?._id)} className='btn btn-sm btn-primary'>
                    <b><i style={{color:'white'}} className="fas fa-user-times me-2"></i>Cancel Request</b>
                    </button>
                    </div>
                    
                
                
                ):(
                    <div className="col">
                    <button  className='btn btn-sm btn-primary'>
                      <b onClick={()=>sendRequest(userDetails?._id)}> <i style={{color:'white'}} className="fa fa-user-plus" aria-hidden="true"></i> Add friend</b>
                    
                    </button>
                    </div>
                    )
                
                ):null
                }

               

                   
                    
                   

    

                </div>
            </div>
            <div className="col d-none d-md-block"></div>
        </div>
        <hr className='mx-auto' style={{width:'70%'}} />
   
        {/* Navigation Tabs */}
        <div className="row mx-0">
            <div className="col-lg-2 d-none d-lg-block"></div>
            <div className="col-auto small" style={profileOption=='posts'?{color:'#0866FF',cursor:'pointer',borderBottom:'1px solid #0866ff'}:{cursor:'pointer'}} onClick={()=>setProfileOption('posts')}>Posts</div>
            <div className="col-auto small">About</div>
            <div className="col-auto small" style={profileOption=='friends'?{color:'#0866FF',cursor:'pointer',borderBottom:'1px solid #0866ff'}:{cursor:'pointer'}} onClick={()=>setProfileOption('friends')}>Friends</div>
            <div className="col-auto small d-none d-sm-inline">Photos</div>
            <div className="col-auto small d-none d-md-inline">Videos</div>
            <div className="col-auto small d-none d-lg-inline">Check</div>
            <div className="col-auto small d-none d-xl-inline">More</div>
            <div className="col d-none d-md-block"></div>
            <div className="col-auto"><i className="fa-solid fa-ellipsis"></i></div>
            <div className="col-lg-2 d-none d-lg-block"></div>
        </div>
       

      {profileOption==='posts'&&(
         <div className="row mx-0 mt-3" style={{backgroundColor:'#F2F4F7', boxShadow:'1px 1px 5px grey'}}>
            <div className="col-lg-2 d-none d-lg-block"></div>
            <div className="col-lg-3 col-md-4 mb-4">
                <div id='profile-intro-div'>
                    <b>Intro</b>
                    <div onClick={()=>console.log(userDetails)} className='small'>
                        <p>
                        Lorem ipsum dolor sit amet consectetur adipisicing elit.
                          </p>
                       
                        
                    </div>
                    <hr />
                    <p className='small'>Studied in Best School System</p>
                    <p className='small'>Went to Garrison Academy Kharian Cantt</p>
                    <p className='small'>Lives in Kharian</p>
                </div>
                <div className='col mt-4' >
                    
                    <div style={{boxShadow:'1px 1px 5px rgba(165, 170, 176, 1)',borderRadius:'10px',backgroundColor:'white'}}>
                        <div style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
                            <h4 className='text-center ms-2 pt-2'>Friends
                                 </h4>
                        <span style={{cursor:'pointer'}}  className='text-primary me-2 ' onClick={()=>setProfileOption('friends')}>See all friends</span>
                        </div>
                        <p className='ms-2 text-muted'>{userRelation?.friends?.length||0} friends</p>
                        <hr />
                        <div   style={{height:'270px',display:'grid',gridTemplateColumns:'1fr 1fr',borderRadius:'10px',overflow:'hidden' }}>
                     
                       
                        {
                            userRelation?.friends?.length>0?(

                             userRelation?.friends?.map((user,index)=>(
                     
                                <Link key={index} to={`/user/${user._id}`} className='ms-3 pt-2 mt-2'  style={{backgroundColor:'#F2F4F7',textDecoration:'none',color:'black',width:'100px',height:'125px',boxShadow:'1px 1px 5px rgba(5, 5, 5, 0.27)',display:'flex',flexDirection:'column',alignContent:'center',borderRadius:'10px',   alignItems: "center"}} >

                                   
                                        {user?.profileImg!=null?(
                                        <img  src={user?.profileImg} width={70} style={{height:'70px',borderRadius:'10px'}} />

                                        ):(
                                             <img  src='/profile.png' width={70} style={{height:'70px',borderRadius:'10px'}} />
                                        )}
                                        <b style={{ margin: 0, textAlign: "center",textDecoration:'none' }}>{user?.name}</b>
                                   
                                </Link> ))
                            ):(
                              
                                <h4 className='text-center ms-4 pt-2' style={{backgroundColor:'white',height:'40px',width:'200px'}}>No Friends</h4>

                              
                            )
                        }
                       
                        
                        </div>
                        </div>

                </div>
            </div>
            
            <div className="col-lg-5 col-md-8 mb-4">
                <div style={{boxShadow:'1px 1px 5px rgba(180, 177, 177, 1)',padding:'2px',marginTop:'9px',borderRadius:'10px', backgroundColor: 'white'}}>
                    <div className="row mt-4 mx-0">
                        <div className="col-1" >
                            <img  className='ms-1' src="/profile.png" width={40} style={{borderRadius:'50%'}} alt="user"/>
                        </div>
                        <div className="col-10">
                            <input className='mt-1 p-2 mb-2 ms-3' type="text" placeholder={`Write something to this ${userDetails?.name}`} id='feed-input' style={{width:'90%',border:'none',backgroundColor:'#E4E6E9',borderRadius:'10px'}} />
                        </div>
                    </div>
                    <hr className='mx-auto' style={{width:'75%'}} />
                    <div className="row mx-0 pb-2 text-center">
                        <div className="col small">Photos</div>
                        <div className="col small">Tag People</div>
                        <div className="col small">Feeling</div>
                    </div>
                </div>




                {userPosts.length>0?(<> 
                
               
                
          
                <div id='post-show-div' style={{backgroundColor:'#F2F4F7'}} >
{userPosts?.map((post,index) => (
   
  <div key={index} className="row mt-3 mb-3"  >
               
               <div   className='mx-auto mt-2 pt-2' style={{width:'95%',boxShadow:'1px 1px 5px grey',padding:'3px',borderRadius:'10px',backgroundColor:'white'}}>
                
               <div className="row pt-2">
              
                <div className="col-1 ms-4 me-2" >
   
                {
                     userDetails?.profileImg!=null?(
                <img src={ userDetails?.profileImg} className='border mt-1' style={{borderRadius:'50%',height:'40px'}}  width={40} />

                    ):(
                <img src="/profile.png" className='border' style={{borderRadius:'50%',height:'40px'}}  width={40} />

                    )
                }
               
                </div>
                <div className="ms-2 col"> <b>{userDetails?.name}</b>
        <p>{new Date(post?.createdAt).toLocaleString()}</p>
       
        </div>
         <hr  style={{color:'grey'}}/>

                     {post?.post?(
                    
                        post?.caption&&(<span className='mx-auto p-2' style={{width:'85%'}}>{post?.caption}</span>)

                     ):(
                       
                        post?.caption&&(<h1 className='mx-auto p-2' style={{width:'85%'}}>{post?.caption}</h1>)

                    
                     )}

         

               </div>
               {post?.post&&(
                  <div style={{display:'flex',justifyContent:'center',alignItems:'center', overflow: 'hidden', borderRadius: '6px'}}>
                <img className='mx-auto' style={{width:'100%',height:'100%',borderRadius:'6px',  objectFit: 'cover'}} src={post?.post} alt="" />

                 </div>
               )}
  
<div  className='pt-1' style={{marginLeft:'40px'}}>   <img src="/like.png" width={20}  /> {post.likes.length} </div>
<hr />
                <div className="row mt-2 mb-2">

                   
                       {
                        post.likes.includes(currentUser._id)?(
                       <div style={{cursor:'pointer'}} onClick={()=>{dislikeFun(post._id)}} className="col-3 ms-5"><i style={{color:'#0866FF'}} className="fa-solid fa-thumbs-up"></i><span className='ms-2'>Like</span></div>
                         ):(
                        <div style={{cursor:'pointer'}} onClick={()=>{likeFun(post._id)}} className="col-3 ms-5"><i  className="fa-regular fa-thumbs-up"></i><span className='ms-2'>Like</span></div>
                        )} 

                        <div className="col-5" onClick={()=>{
                              setComment(true);
                        //   setSelectedPost(post)
                        }  } style={{cursor:'pointer'}}><i className="fa-regular fa-comment"></i>Comment</div>

                        <div className="col"><img src="/share.png" width={17} />Share</div>

                </div>
               </div>

                </div>   

     
))}

 </div>

 </>):(
    <div className='mt-5'style={{boxShadow:'1px 1px 5px #afafb3ff',borderRadius:'10px',backgroundColor:'white'}}>
<h1 className='p-5 text-center text-muted'>No Post Yet</h1>
 </div>)}


  </div>

<div className="col-lg-2 d-none d-lg-block"></div>
        </div> ) }

{profileOption==='friends'&&(<Friends allFriends={userRelation?.friends}/>)}
      
        </div>

)}

export default Profile