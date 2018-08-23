use ndarray::prelude::*;
use std::convert::{From, TryInto};

#[derive(Clone, Abomonation, Eq, PartialEq, Debug, PartialOrd, Ord)]
pub struct AbomonableArray<A, D> {
    data: Vec<A>,
    strides: D,
    shape: D,
}

pub type AbomonableArray1<A> = AbomonableArray<A, [usize; 1]>;
pub type AbomonableArray2<A> = AbomonableArray<A, [usize; 2]>;
pub type AbomonableArray3<A> = AbomonableArray<A, [usize; 3]>;
pub type AbomonableArray4<A> = AbomonableArray<A, [usize; 4]>;
pub type AbomonableArray5<A> = AbomonableArray<A, [usize; 5]>;
pub type AbomonableArray6<A> = AbomonableArray<A, [usize; 6]>;
pub type AbomonableArrayDyn<A> = AbomonableArray<A, Vec<usize>>;

impl<A> From<Array<A, IxDyn>> for AbomonableArray<A, Vec<usize>> {
    #[inline]
    fn from(from: Array<A, IxDyn>) -> Self {
        let shape = from.shape().to_vec();
        let strides = from.strides().iter().map(|n| *n as usize).collect();
        AbomonableArray {
            data: from.into_raw_vec(),
            strides,
            shape,
        }
    }
}

impl<A> Into<Array<A, IxDyn>> for AbomonableArray<A, Vec<usize>> {
    #[inline]
    fn into(self) -> Array<A, IxDyn> {
        <Array<A, IxDyn>>::from_shape_vec(self.shape.strides(self.strides), self.data).unwrap()
    }
}

macro_rules! impl_array {
    ($index:ty) => {
        impl<A> From<Array<A, Dim<$index>>> for AbomonableArray<A, $index> {
            #[inline]
            fn from(from: Array<A, Dim<$index>>) -> Self {
                let shape: $index = {
                    let slice: &$index = from.shape().try_into().unwrap();
                    slice.clone()
                };
                let strides: $index = {
                    let slice: &$index = unsafe {
                        &*(from.strides() as *const [isize] as *const [usize])
                    }.try_into()
                        .unwrap();
                    slice.clone()
                };

                AbomonableArray {
                    data: from.into_raw_vec(),
                    strides,
                    shape,
                }
            }
        }

        impl<A> Into<Array<A, Dim<$index>>> for AbomonableArray<A, $index> {
            #[inline]
            fn into(self) -> Array<A, Dim<$index>> {
                <Array<A, Dim<$index>>>::from_shape_vec(
                    self.shape.strides(self.strides), self.data
                ).unwrap()
            }
        }
    };
}

impl_array!([usize; 1]);
impl_array!([usize; 2]);
impl_array!([usize; 3]);
impl_array!([usize; 4]);
impl_array!([usize; 5]);
impl_array!([usize; 6]);

macro_rules! impl_into_array_view {
    ($index:ident, $abom_type:ident) => {
        impl<'a, A> Into<ArrayView<'a, A, $index>> for &'a $abom_type<A>
        where
            A: 'a,
        {
            #[inline]
            fn into(self) -> ArrayView<'a, A, $index> {
                <ArrayView<'a, A, $index>>::from_shape(
                    self.shape.clone().strides(self.strides.clone()),
                    self.data.as_slice(),
                ).unwrap()
            }
        }
    };
}

macro_rules! impl_into_array_view_mut {
    ($index:ident, $abom_type:ident) => {
        impl<'a, A> Into<ArrayViewMut<'a, A, $index>> for &'a mut $abom_type<A>
        where
            A: 'a,
        {
            #[inline]
            fn into(self) -> ArrayViewMut<'a, A, $index> {
                <ArrayViewMut<'a, A, $index>>::from_shape(
                    self.shape.clone().strides(self.strides.clone()),
                    self.data.as_mut_slice(),
                ).unwrap()
            }
        }
    };
}

impl_into_array_view!(Ix1, AbomonableArray1);
impl_into_array_view!(Ix2, AbomonableArray2);
impl_into_array_view!(Ix3, AbomonableArray3);
impl_into_array_view!(Ix4, AbomonableArray4);
impl_into_array_view!(Ix5, AbomonableArray5);
impl_into_array_view!(Ix6, AbomonableArray6);
impl_into_array_view!(IxDyn, AbomonableArrayDyn);

impl_into_array_view_mut!(Ix1, AbomonableArray1);
impl_into_array_view_mut!(Ix2, AbomonableArray2);
impl_into_array_view_mut!(Ix3, AbomonableArray3);
impl_into_array_view_mut!(Ix4, AbomonableArray4);
impl_into_array_view_mut!(Ix5, AbomonableArray5);
impl_into_array_view_mut!(Ix6, AbomonableArray6);
impl_into_array_view_mut!(IxDyn, AbomonableArrayDyn);

pub trait AsView<'a, D, T> {
    fn view(self) -> ArrayView<'a, T, D>;
}

pub trait AsMutView<'a, D, T> {
    fn view_mut(self) -> ArrayViewMut<'a, T, D>;
}

impl<'a, A, D, T: 'a> AsView<'a, D, T> for A
where
    A: Into<ArrayView<'a, T, D>>,
{
    #[inline]
    fn view(self) -> ArrayView<'a, T, D> {
        self.into()
    }
}

impl<'a, A, D, T: 'a> AsMutView<'a, D, T> for A
where
    A: Into<ArrayViewMut<'a, T, D>>,
{
    #[inline]
    fn view_mut(self) -> ArrayViewMut<'a, T, D> {
        self.into()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use abomonation::{decode, encode};

    #[test]
    fn test_as_view() {
        let a = arr2(&[[1, 2, 3], [4, 5, 6]]);
        let a_backup = a.clone();
        let a2: AbomonableArray<_, _> = a.into();
        let a_view = a2.view();
        assert_eq!(a_view, a_backup)
    }

    #[test]
    fn test_conversion() {
        let a = arr2(&[[1, 2, 3], [4, 5, 6]]);
        let a_backup = a.clone();
        let a2: AbomonableArray<_, _> = a.into();
        let a3: Array2<_> = a2.into();
        assert_eq!(a_backup, a3);
    }

    #[test]
    fn test_array_abomonate() {
        let a = arr2(&[[1, 2, 3], [4, 5, 6]]);
        let a_clone = a.clone();
        let a2: AbomonableArray<_, _> = a.into();

        let mut bytes = Vec::new();
        unsafe { encode(&a2, &mut bytes).unwrap() };

        if let Some((result, remaining)) = unsafe { decode::<AbomonableArray<_, _>>(&mut bytes) } {
            let result_array: ArrayView2<usize> = result.into();
            assert_eq!(result_array, a_clone);
            assert_eq!(remaining.len(), 0);
        }
    }
}
