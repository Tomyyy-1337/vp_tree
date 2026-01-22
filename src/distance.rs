
/// [`Distance`] trait to be implemented by types stored in the [`crate::VpTree`].
/// The [`Distance::distance`] method should return a non-negative [f64] representing the distance between self and other.
/// Elements in the tree have to implement [`Distance`] to themselves. Additionally, search targets can implement [`Distance`] to the stored type.
/// 
/// ## Example 1
/// ```rust
/// use vp_tree::Distance;
/// 
/// struct Point {
///     x: f64,
///     y: f64,
/// }
/// 
/// impl Distance<Point> for Point {
///     fn distance(&self, other: &Point) -> f64 {
///         ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
///     }
/// }
/// ```
/// 
/// The [`Point`] struct in the first example does not store any additional data, we use the same struct for both storage and search target.
/// 
/// ## Example 2 
/// ```rust
/// use vp_tree::Distance;
/// 
/// struct DataPoint {
///    point: Point,
///    data: String,
/// }
/// impl Distance<DataPoint> for DataPoint {
///    fn distance(&self, other: &DataPoint) -> f64 {
///       ((self.point.x - other.point.x).powi(2) + (self.point.y - other.point.y).powi(2)).sqrt()
///   }
/// }
/// struct Point {
///    x: f64,
///    y: f64,
/// }
/// impl Distance<DataPoint> for Point {
///     fn distance(&self, other: &DataPoint) -> f64 {
///        ((self.x - other.point.x).powi(2) + (self.y - other.point.y).powi(2)).sqrt()
///    }
/// }    
/// ```
/// The second example shows a `DataPoint` struct that stores additional data alongside the point coordinates.
/// The `DataPoint` struct implements [`Distance`] to itself to enable storage in the [`crate::VpTree`]. 
/// Additionally, the `Point` struct implements [`Distance`] to `DataPoint`, allowing it to be used as a search target without storing additional unnecessary data.
pub trait Distance<T> {
    /// Metric distance between self and other. Should be non-negative. Squared distances do not work. 
    fn distance(&self, other: &T) -> f64;
}