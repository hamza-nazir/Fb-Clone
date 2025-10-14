import { useEffect,useState,useRef} from 'react';
import { useContext } from 'react';
import { Context } from '../Hook/Context';
import Comment from './Comment';
import useAuth from '../Hook/useAuth';
import { Link } from 'react-router-dom';
import api from '../Hook/Axios'
const Posts = () => {
        const postDiv=useRef();
        const{setComment,posts,setSelectedPost,setPosts}=useContext(Context);
        const [page,setPage]=useState(1);
         const currentUser=  useAuth();

        useEffect(()=>{
      
         postDiv?.current?.addEventListener('scroll',()=>{
          if(postDiv?.current?.clientHeight+postDiv?.current?.scrollTop+1>postDiv?.current?.scrollHeight)
            { setPage(curr=>curr+1)}      
         })         
        },[posts])




        useEffect(()=>{
        //  console.log("E#d")
        api.post('/user-feed',{page})
        .then((res)=>{
            setPosts(curr => [...curr, ...res.data.certifedPosts]);
       
        })
        .catch((err)=>{
            console.log(err);
        })
        },[page])
  
  function likeFun(postID){
api.post('/like',{postID})
    .then((res)=>{
        const updatedPost=res.data.postToLike;
        setPosts(prevPosts =>
  prevPosts.map(post =>
    post._id === updatedPost._id ? updatedPost : post
  )
);
})
}


function dislikeFun(postID){
api.post('/dislike',{postID})
    .then((res)=>{
        const dislikedPost=res.data.postTodislike;
        // console.log(dislikedPost);
        setPosts(prevPosts=>
            prevPosts.map(post => 
             post._id === dislikedPost._id ? dislikedPost : post
            )
        )
    })
}




    return (
        <>
       <Comment  />

{
  posts?.length>0?(


        <div className='container-fluid' ref={postDiv} style={{ height: '92vh', overflowY: 'scroll',backgroundColor:'#F2F4F7' }} >
            {

                posts?.map((post,index) => (
                
                    <div key={index} className="row mb-3 bg-light rounded shadow-sm" >
                      <Link  to={`/user/${post.owner._id}`} style={{cursor:'pointer',textDecoration:'none',color:'black'}} className='row'>
                          <div className="col-2 mt-2">
                            {post?.owner?.profileImg?(
                            <img src={post?.owner?.profileImg} width={50} style={{borderRadius:'50%',height:'50px'}} />

                            ):(
                            <img  src='/profile.png' width={50} style={{borderRadius:'50%',height:'50px'}} />
                              
                            )}
                          </div>
                        <div className="col-10 mt-3">
                            <div className="row">
                                <div className="col-12"> <b>{post.owner.name}</b></div>
                               <div className="col text-muted">{new Date(post.createdAt).toLocaleString()}</div>
                            </div>
                           
                            
                            </div>
                        </Link>
                        {/* <button onClick={()=>console.log(posts)}>de</button> */}
                             {post.post?(
                      <span > {post.caption}</span>

                             ):(
                      <h1 > {post.caption}</h1>
                              
                             )
                             }
                     
                      {post.post&&(
                        <>
                        <hr />
                          <div>
                       <img src={post.post} style={{width:'100%'}} />
                      </div>
                        </>
                      
                      )}
                      <div className='mt-2 pb-1' style={{display:'flex',alignItems:'center',flexDirection:'row',borderBottom:'1px solid #C2C4C6'}}>
                        <img src="/like.png" width={20} className='me-1' />
                       <div style={{display:'inline-flex'}} className='text-muted'> {post?.likes.length}</div>
                        {/* <hr className='mt-2'/> */}
                      </div>
                   
                     <div className="container-fluid mt-2">
                         <div className="row pb-2">
               

                     {
  post.likes.some(user => user === currentUser._id)
    ? (
      <div style={{cursor:'pointer'}} onClick={() => dislikeFun(post._id)} className="col ps-4">
        <i style={{color:'#0866FF'}} className="fa-solid fa-thumbs-up"></i> Like
      </div>
    )
    : (
      <div style={{cursor:'pointer'}} onClick={() => likeFun(post._id)} className="col ps-4">
        <i className="fa-regular fa-thumbs-up"></i> Like
      </div>
    )
}



                        <div className="col-5" style={{cursor:'pointer'}} onClick={()=>{
                          setComment(true);
                          setSelectedPost(post)
                          }}><i className="fa-regular fa-comment"></i>Comment</div>
                        <div className="col"><img src="/share.png" width={17} />Share</div>
                      </div>
                     </div>
                    </div>
                  
                ))
            }
        </div>
  ):(
  <div className='rounded-2' style={{backgroundColor:'white',height:'140px',display:'flex',justifyContent:'center',alignItems:'center'}}>
      <h3 className='text-center text-muted'>
  No posts available. Your friends haven’t shared anything yet.
</h3>

  </div>
  )
}
        </>
    )

}

export default Posts
