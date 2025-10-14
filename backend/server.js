const express = require('express');
const session = require('express-session');
const Post=require('./models/Post');
const Comment=require('./models/Comment')
const Relation=require('./models/Relation')
const passport=require('passport')
const LocalStrategy=require('passport-local');
const multer  = require('multer')
const { storage }=require('./cloudinary'); 
const upload = multer({ storage });
// const upload = multer({ dest: 'uploads/' })
const User=require('./models/User')
const app = express();
const cors=require('cors')


app.use(cors({
    origin:'http://localhost:5173',
    credentials:true
}))



app.use(express.json());
app.use(express.urlencoded({extended: true}));
app.use(session({
  secret: 'keyboardCat',
  resave: false,
  saveUninitialized: true,
  cookie: {
    secure: false,
    maxAge:1000*60*60*24*7,
    httpOnly:true
   }
}));


app.use(passport.initialize());
app.use(passport.session());
passport.use(new LocalStrategy({ usernameField: 'email' }, User.authenticate()));
passport.serializeUser(User.serializeUser());
passport.deserializeUser(User.deserializeUser());

const protectedRoute=(req,res,next)=>{
  if(!req.isAuthenticated()){
    return res.json({success:false})
  }
  next();
}

app.post('/auth',(req,res)=>{
  if(!req.isAuthenticated()){
    res.json({success:false})
  }else{
       res.json({success:true,user:req.user})
  }
})


app.post('/signup', async (req, res) => {
  try{
const {name,surname,email,password,gender}=req.body;
let user=await User.register({name,surname,email,gender},password);
if(user){
req.logIn(user,()=>{
    const {email,gender,name,surname,_id}=req.user;
res.json({success:true,user: {email,gender,name,surname,id:_id}})
})
}
  }catch(err){
    res.json({success:false,msg:'Email already Exist'});
  }

});


app.post('/login', passport.authenticate('local'), (req, res) => {
  const {email,gender,name,surname,_id}=req.user;
  res.json({ message: 'Login successful', user: {email,gender,name,surname,id:_id} });
});


app.post('/get-user',async(req,res)=>{
const {searchUser}=req.body;
// console.log(searchUser);
 const user = await User.find({
      name: { $regex: searchUser, $options: 'i' } 
    });

   if(user){
    res.json({user})
   }else{
    res.json({success:false})
   }
})


app.post('/profile-view',protectedRoute,async(req,res)=>{
let {id}=req.body;
const user=await User.findById(id);
const userPosts = await Post.find({ owner: user._id });
const userRelationDocID=user._id;
let sendRequestDocument= await Relation.findOne({owner:userRelationDocID}) 
.populate('friends')   
  .populate('requests');
 res.json({user,userPosts,sendRequestDocument});

})

app.post('/upload-data',upload.single('imageFile'),async (req,res)=>{

 const {caption}=req.body;
   const post=new Post({post:req.file.path,caption,owner:req.user._id})
   await post.save();
   console.log("OK");
   res.json({success:true})

 
  
})

app.post('/upload-caption',async(req,res)=>{
  const {caption}=req.body;
  const post=new Post({caption,owner:req.user._id})
   await post.save();
     res.json({success:true})
})


app.post('/logout',(req,res)=>{
  req.logOut(()=>{
    res.json({success:true})
  })
})

app.post('/like',protectedRoute,async(req,res)=>{
  const {postID}=req.body;
 const postToLike= await Post.findOne({_id:postID});
if(!postToLike.likes.includes(req.user._id)){
postToLike.likes.push(req.user._id);
await postToLike.save();
await postToLike.populate('owner')
res.json({postToLike})
}

})

app.post('/dislike',protectedRoute,async(req,res)=>{
  const {postID}=req.body;
const postTodislike= await Post.findOne({_id:postID});
if(postTodislike.likes.includes(req.user._id)){
postTodislike.likes.pull(req.user._id);

await postTodislike.save();
await postTodislike.populate('owner')
res.json({ postTodislike });
}

})

app.post('/send-request',async(req,res)=>{
  const {userToSendRequestID}=req.body;
  const currentUserDoc=await Relation.findOne({owner:req.user._id})
  let sendRequestDocument= await Relation.findOne({owner:userToSendRequestID})
    if(!sendRequestDocument){
        sendRequestDocument=new Relation({owner:userToSendRequestID})
    }
    if (!sendRequestDocument?.requests?.some(id => id.toString() === req.user._id.toString()) &&sendRequestDocument?.owner?.toString()!=req.user._id.toString()
    && !currentUserDoc?.friends?.some(id => id.toString() === userToSendRequestID.toString())
    ) {
       sendRequestDocument.requests.push(req.user._id);
    await sendRequestDocument.save();
    await sendRequestDocument.populate('friends');
  await sendRequestDocument.populate('requests');
  res.json({sendRequestDocument});
    }else{
         await sendRequestDocument.populate('friends');
  res.json({sendRequestDocument});

    }
    
    
})


app.post('/cancel-request',async(req,res)=>{
const {userToCancelRequestID}=req.body;
let cancelRequestDocument= await Relation.findOne({owner:userToCancelRequestID});
cancelRequestDocument.requests.pull(req.user._id)
await cancelRequestDocument.save();
  await cancelRequestDocument.populate('friends');
  await cancelRequestDocument.populate('requests');
res.json({cancelRequestDocument});
})



app.post('/get-request',async(req,res)=>{
try{
let checkRequestDocument= await Relation.findOne({owner:req.user._id}).populate('requests')
// console.log(checkRequestDocument);
res.json({checkRequestDocument})
}catch(err){
  console.log("ERROR");
}
})

app.post('/accept-request',async(req,res)=>{
  const {userToAcceptRequest}=req.body;
const currentUserDoc= await Relation.findOne({owner:req.user._id}).populate('requests').populate('friends')
currentUserDoc.friends.push(userToAcceptRequest);
currentUserDoc.requests.pull(userToAcceptRequest)
await currentUserDoc.save();

const userToAcceptRequestDoc=await Relation.findOne({owner:userToAcceptRequest})
if(!userToAcceptRequestDoc){
  const tempObj=new Relation({
    owner:userToAcceptRequest,
    friends:[req.user._id]
  })
  await tempObj.save();
}
console.log(currentUserDoc);
res.json({currentUserDoc})
})



app.post('/edit-user',upload.fields([ { name: "profileImg", maxCount: 1 },{ name: "coverImg", maxCount: 1 }]),async(req,res)=>{
try{


   let coverPhoto=null;
      let profilePhoto=null;

   
    if (req.files && req.files["coverImg"] && req.files["coverImg"][0]) {
      coverPhoto = req.files["coverImg"][0].path;
    }

    if (req.files && req.files["profileImg"] && req.files["profileImg"][0]) {
      profilePhoto = req.files["profileImg"][0].path;
    }

    const user=await User.findById(req.user._id);
  
    user.profileImg=profilePhoto;


user.coverImg=coverPhoto;
   
    await user.save();
   res.json({user})}
   catch (err) {
      console.error("Upload failed");
    }
  }
);

app.post("/user-feed",protectedRoute,async(req,res)=>{

 if(req.user._id){
  const {page}=req.body;
  console.log(page)
  const skip=(page-1)*10
  let currrentUserID=req.user._id;
  const currentUserDoc=await Relation?.findOne({owner:currrentUserID});
  const friendsArray=currentUserDoc?.friends;
  const certifedPosts=await Post.find({owner:{$in:friendsArray}})
  .populate("owner")
  .skip(skip)
  .limit(10)
  res.json({certifedPosts});
  }
})




app.post('/comment',async(req,res)=>{
  if(req.user._id){
    const {post,cmnt}=req.body;
 const cmntObj=new Comment({postID:post._id,comment:cmnt,owner:req.user._id});
 await cmntObj.save();
const cmntData= await Comment.find({postID:post._id})
 .populate('postID')
  .populate('owner');
res.json({cmntData});

  }
  
})


app.post('/get-comments', async (req, res) => {
  try {
    const { postID } = req.body;
    if (!postID) return res.status(400).json({ error: "Post ID required" });

    const cmntData = await Comment.find({ postID })
      .populate('owner')
      .sort({ createdAt: -1 });

    res.json({ comments: cmntData });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Server error" });
  }
});
app.listen(3000, () => {
console.log('App listening on port');
});